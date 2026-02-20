#!/usr/bin/env python3
"""
Multi-step parallel execution script
Supports GPU allocation and step selection
"""

import os
import sys
import time
import argparse
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import logging
import tempfile
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineExecutor:
    def __init__(self, csv_directory, video_directory, output_base_dir="eval_outputs",
                 original_videos_path=None, original_masks_path=None):
        self.csv_directory = csv_directory
        self.video_directory = video_directory
        self.output_base_dir = output_base_dir
        self.original_videos_path = original_videos_path
        self.original_masks_path = original_masks_path

        # Ensure output directory exists
        os.makedirs(self.output_base_dir, exist_ok=True)

        # Set environment variables to ensure scripts can find modules
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

    def set_gpu_env(self, gpu_id):
        """Set GPU environment variables"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        return env

    def create_temp_config(self, base_config_path, output_subdir, original_videos_path=None, original_masks_path=None):
        """
        Create a temporary config file using externally specified path parameters

        Args:
            base_config_path: Base config file path (for obtaining fixed parameters like model paths)
            output_subdir: Output subdirectory name
            original_videos_path: Original videos path (optional)
            original_masks_path: Original masks path (optional)

        Returns:
            Path to the temporary config file
        """
        # Read base config file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_config_full_path = os.path.join(script_dir, base_config_path)

        if os.path.exists(base_config_full_path):
            with open(base_config_full_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # If base config file doesn't exist, create default config
            config = {}
            logger.warning(f"Base config file not found: {base_config_full_path}, using default config")

        # Update path parameters in the config
        config['csv_directory'] = self.csv_directory
        config['edit_root_path'] = self.video_directory
        config['output_path'] = os.path.join(self.output_base_dir, output_subdir)

        # If original video and masks paths are provided, update config
        if original_videos_path:
            config['original_videos_path'] = original_videos_path
        if original_masks_path:
            config['original_masks_path'] = original_masks_path

        # Ensure use_mask parameter exists
        if 'use_mask' not in config:
            config['use_mask'] = True

        # Create temporary config file
        temp_dir = tempfile.gettempdir()
        temp_config_path = os.path.join(temp_dir, f"temp_{output_subdir}_config.yaml")

        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Created temporary config file: {temp_config_path}")
        logger.debug(f"Config content: {config}")

        return temp_config_path

    def run_step1_extract_masks(self, gpu_id=0):
        """Step 1: Extract masks (GPU 0)"""
        logger.info(f"Starting step 1: extract_masks_output.py (GPU {gpu_id})")

        try:
            # Import module and execute
            env = self.set_gpu_env(gpu_id)

            # Use subprocess to run, ensuring GPU isolation
            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, r'{os.path.dirname(os.path.abspath(__file__))}')
from extract_masks_output import process_videos_in_csv_directory

result = process_videos_in_csv_directory(
    r'{self.csv_directory}',
    r'{self.video_directory}',
    r'{self.video_directory}'
)
print(f'Step 1 completed: processed {{result}} videos')
"""
            ]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                logger.info(f"Step 1 completed: {result.stdout.strip()}")
                return True, result.stdout.strip()
            else:
                logger.error(f"Step 1 failed: {result.stderr}")
                return False, result.stderr

        except Exception as e:
            logger.error(f"Step 1 execution error: {str(e)}")
            return False, str(e)

    def run_similarity_script(self, script_name, output_subdir, gpu_id=1):
        """Run a similarity script"""
        logger.info(f"Starting: {script_name} (GPU {gpu_id})")

        try:
            output_dir = os.path.join(self.output_base_dir, output_subdir)
            env = self.set_gpu_env(gpu_id)

            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, r'{os.path.dirname(os.path.abspath(__file__))}')
from {script_name.replace('.py', '')} import process_videos_in_csv_directory

result = process_videos_in_csv_directory(
    r'{self.csv_directory}',
    r'{self.video_directory}',
    r'{output_dir}'
)
print(f'{script_name} completed: processed {{len(result)}} videos')
"""
            ]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                logger.info(f"{script_name} completed: {result.stdout.strip()}")
                return True, result.stdout.strip()
            else:
                logger.error(f"{script_name} failed: {result.stderr}")
                return False, result.stderr

        except Exception as e:
            logger.error(f"{script_name} execution error: {str(e)}")
            return False, str(e)

    def run_step2_similarities(self, gpu_id=1):
        """Step 2: Run three similarity scripts (parallel, GPU 1)"""
        logger.info(f"Starting step 2: similarity analysis scripts (GPU {gpu_id})")

        scripts = [
            ("text_similarity.py", "text_similarity"),
            ("temporal_consistency.py", "temporal_consistency"),
            ("appearance_consistency.py", "appearance_consistency")
        ]

        results = {}

        # Use ThreadPoolExecutor to run three scripts in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_script = {
                executor.submit(self.run_similarity_script, script, output_dir, gpu_id): script
                for script, output_dir in scripts
            }

            for future in as_completed(future_to_script):
                script = future_to_script[future]
                try:
                    success, result = future.result()
                    results[script] = (success, result)
                except Exception as e:
                    logger.error(f"Exception while executing {script}: {str(e)}")
                    results[script] = (False, str(e))

        # Check if all scripts succeeded
        all_success = all(success for success, _ in results.values())

        if all_success:
            logger.info("Step 2 fully completed")
        else:
            logger.warning("Step 2 partially failed")

        return all_success, results

    def run_fidelity_script(self, script_path, config_path, gpu_id):
        """Run a fidelity script"""
        script_name = os.path.basename(script_path)
        logger.info(f"Starting: {script_name} (GPU {gpu_id})")

        try:
            env = self.set_gpu_env(gpu_id)

            cmd = [
                sys.executable,
                script_path,
                "--config_path", config_path
            ]

            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)  # 2 hour timeout

            if result.returncode == 0:
                logger.info(f"{script_name} completed: {result.stdout.split('Overall')[-1].strip() if 'Overall' in result.stdout else 'Success'}")
                return True, result.stdout
            else:
                logger.error(f"{script_name} failed: {result.stderr}")
                return False, result.stderr

        except Exception as e:
            logger.error(f"{script_name} execution error: {str(e)}")
            return False, str(e)

    def run_step3_fidelity(self, gpu_ids=[0, 1], original_videos_path=None, original_masks_path=None):
        """
        Step 3: Run two fidelity scripts (parallel, GPU 0 and 1)

        Args:
            gpu_ids: List of GPU device IDs
            original_videos_path: Original videos path (optional)
            original_masks_path: Original masks path (optional)
        """
        logger.info(f"Starting step 3: fidelity analysis scripts (GPU {gpu_ids})")

        # Script configuration
        scripts_config = [
            (
                "flow_fidelity/flow_fidelity_multi.py",
                "flow_fidelity/configs/flow_fidelity_score_config_multi.yaml",
                "flow_fidelity",
                gpu_ids[0]
            ),
            (
                "motion_fidelity/motion_fidelity_multi.py",
                "motion_fidelity/configs/motion_fidelity_score_config_multi.yaml",
                "motion_fidelity",
                gpu_ids[1] if len(gpu_ids) > 1 else gpu_ids[0]
            )
        ]

        # Generate temporary config files
        temp_configs = []
        script_dir = os.path.dirname(os.path.abspath(__file__))

        for script_path, base_config_path, output_subdir, gpu_id in scripts_config:
            temp_config_path = self.create_temp_config(
                base_config_path,
                output_subdir,
                original_videos_path,
                original_masks_path
            )
            temp_configs.append(temp_config_path)

        results = {}

        # Use ThreadPoolExecutor to run two scripts in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_script = {}

            for i, (script_path, base_config_path, output_subdir, gpu_id) in enumerate(scripts_config):
                script_full_path = os.path.join(script_dir, script_path)
                future = executor.submit(
                    self.run_fidelity_script,
                    script_full_path,
                    temp_configs[i],
                    gpu_id
                )
                future_to_script[future] = script_path

            for future in as_completed(future_to_script):
                script_path = future_to_script[future]
                script_name = os.path.basename(script_path)
                try:
                    success, result = future.result()
                    results[script_name] = (success, result)
                except Exception as e:
                    logger.error(f"Exception while executing {script_name}: {str(e)}")
                    results[script_name] = (False, str(e))

        # Clean up temporary config files
        for temp_config in temp_configs:
            try:
                if os.path.exists(temp_config):
                    os.remove(temp_config)
                    logger.debug(f"Deleted temporary config file: {temp_config}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary config file: {temp_config}, error: {e}")

        # Check if all scripts succeeded
        all_success = all(success for success, _ in results.values())

        if all_success:
            logger.info("Step 3 fully completed")
        else:
            logger.warning("Step 3 partially failed")

        return all_success, results

    def run_pipeline(self, steps_to_run=None):
        """Run the full pipeline or specified steps"""
        if steps_to_run is None:
            steps_to_run = [1, 2, 3]

        logger.info(f"Starting pipeline, steps: {steps_to_run}")
        start_time = time.time()

        step_results = {}
        step_futures = {}

        # Use ThreadPoolExecutor for parallel execution between steps
        with ThreadPoolExecutor(max_workers=3) as executor:

            # Step 1 and Step 2 can run in parallel
            if 1 in steps_to_run:
                step_futures[1] = executor.submit(self.run_step1_extract_masks, 0)

            if 2 in steps_to_run:
                step_futures[2] = executor.submit(self.run_step2_similarities, 1)

            # Wait for Step 1 and Step 2 to complete
            for step_num in [1, 2]:
                if step_num in step_futures:
                    try:
                        success, result = step_futures[step_num].result()
                        step_results[step_num] = (success, result)
                        logger.info(f"Step {step_num} completed, status: {'success' if success else 'failed'}")
                    except Exception as e:
                        logger.error(f"Step {step_num} execution error: {str(e)}")
                        step_results[step_num] = (False, str(e))

            # Step 3 needs to wait for previous steps to complete
            if 3 in steps_to_run:
                logger.info("Starting step 3...")
                success, result = self.run_step3_fidelity(
                    [0, 1],
                    self.original_videos_path,
                    self.original_masks_path
                )
                step_results[3] = (success, result)
                logger.info(f"Step 3 completed, status: {'success' if success else 'failed'}")

        # Summarize results
        total_time = time.time() - start_time
        successful_steps = [step for step, (success, _) in step_results.items() if success]
        failed_steps = [step for step, (success, _) in step_results.items() if not success]

        logger.info(f"Pipeline execution completed! Total time: {total_time/60:.1f} minutes")
        logger.info(f"Successful steps: {successful_steps}")
        if failed_steps:
            logger.warning(f"Failed steps: {failed_steps}")

        return step_results


def main():
    """
    python eval2/run_pipeline.py \
    --csv_directory "benchmark_new/captions_inf_all" \
    --original_videos_path "benchmark_new/reference_videos" \
    --original_masks_path "benchmark_new/reference_video_masks_eval" \
    --video_directory "outputs_mask_fixed_token_nlastframe2_k15_wodme_tarwomask" \
    --output_directory "eval_outputs_mask_fixed_token_nlastframe2_k15_wodme_tarwomask"
    """
    parser = argparse.ArgumentParser(description="Multi-step parallel execution script")
    parser.add_argument("--csv_directory", type=str, required=True,
                       help="CSV file directory path")
    parser.add_argument("--video_directory", type=str, required=True,
                       help="Video file directory path")
    parser.add_argument("--output_directory", type=str, default="eval_outputs",
                       help="Output directory path")
    parser.add_argument("--original_videos_path", type=str, default=None,
                       help="Original videos path (for step 3 fidelity analysis)")
    parser.add_argument("--original_masks_path", type=str, default=None,
                       help="Original masks path (for step 3 fidelity analysis)")
    parser.add_argument("--steps", type=str, default="1,2,3",
                       help="Steps to execute, comma-separated (e.g.: 1,2,3 or 1,3)")
    parser.add_argument("--step1", action="store_true",
                       help="Execute step 1 only: extract_masks_output.py")
    parser.add_argument("--step2", action="store_true",
                       help="Execute step 2 only: three similarity scripts")
    parser.add_argument("--step3", action="store_true",
                       help="Execute step 3 only: two fidelity scripts")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.csv_directory):
        logger.error(f"CSV directory not found: {args.csv_directory}")
        return 1

    if not os.path.exists(args.video_directory):
        logger.error(f"Video directory not found: {args.video_directory}")
        return 1

    # Parse steps to execute
    steps_to_run = []

    if args.step1:
        steps_to_run.append(1)
    if args.step2:
        steps_to_run.append(2)
    if args.step3:
        steps_to_run.append(3)

    # If no individual steps specified, use --steps parameter
    if not steps_to_run:
        try:
            steps_to_run = [int(x.strip()) for x in args.steps.split(',')]
        except ValueError:
            logger.error("Steps parameter format error, should be comma-separated numbers, e.g.: 1,2,3")
            return 1

    # Validate step range
    valid_steps = [step for step in steps_to_run if step in [1, 2, 3]]
    if not valid_steps:
        logger.error("No valid steps, valid steps are: 1, 2, 3")
        return 1

    logger.info(f"Configuration:")
    logger.info(f"  CSV directory: {args.csv_directory}")
    logger.info(f"  Video directory: {args.video_directory}")
    logger.info(f"  Output directory: {args.output_directory}")
    if args.original_videos_path:
        logger.info(f"  Original videos path: {args.original_videos_path}")
    if args.original_masks_path:
        logger.info(f"  Original masks path: {args.original_masks_path}")
    logger.info(f"  Steps to execute: {valid_steps}")

    # Create executor and run
    executor = PipelineExecutor(
        csv_directory=args.csv_directory,
        video_directory=args.video_directory,
        output_base_dir=args.output_directory,
        original_videos_path=args.original_videos_path,
        original_masks_path=args.original_masks_path
    )

    try:
        results = executor.run_pipeline(valid_steps)

        # Output detailed results
        print("\n" + "="*60)
        print("Execution Results Summary:")
        print("="*60)

        for step, (success, result) in results.items():
            status = "Success" if success else "Failed"
            print(f"Step {step}: {status}")
            if not success:
                print(f"  Error: {result}")

        # Return appropriate exit code
        if all(success for success, _ in results.values()):
            print("\nAll steps executed successfully!")
            return 0
        else:
            print("\nSome steps failed, check logs for details.")
            return 1

    except KeyboardInterrupt:
        logger.info("User interrupted execution")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during execution: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
