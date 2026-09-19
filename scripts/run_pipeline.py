"""
CLI entry point for Resume Matcher.

Usage:
    python -m scripts.run_pipeline --jd path/to/jd.pdf --resume path/to/resume.pdf
    python -m scripts.run_pipeline --jd path/to/jd.pdf --resume-dir path/to/resumes/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from resume_matcher.config import load_config, setup_logging
from resume_matcher.services.pipeline_service import PipelineService


def main() -> None:
    """Main entry point for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Resume-JD Matching Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single resume
  python -m scripts.run_pipeline --jd jd.pdf --resume resume.pdf

  # Multiple resumes (directory)
  python -m scripts.run_pipeline --jd jd.pdf --resume-dir ./resumes/

  # Custom config and output
  python -m scripts.run_pipeline --jd jd.pdf --resume resume.pdf \\
      --config config/custom.yaml --output ./reports/
        """,
    )
    parser.add_argument(
        "--jd",
        required=True,
        help="Path to the job description file (PDF or DOCX)",
    )
    parser.add_argument(
        "--resume",
        help="Path to a single resume file (PDF or DOCX)",
    )
    parser.add_argument(
        "--resume-dir",
        help="Path to a directory of resume files",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory for reports (default: output)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.resume and not args.resume_dir:
        parser.error("Either --resume or --resume-dir is required")

    if args.resume and args.resume_dir:
        parser.error("Specify either --resume or --resume-dir, not both")

    # Load configuration
    config = load_config(config_path=args.config)
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Resume Matcher Pipeline starting")

    # Initialize pipeline
    pipeline = PipelineService(config)

    try:
        if args.resume:
            # Single resume mode
            report_path = pipeline.process_single(
                jd_path=args.jd,
                resume_path=args.resume,
                output_dir=args.output,
            )
            logger.info("Report generated: %s", report_path)
            print(f"\n✅ Report generated: {report_path}")

        elif args.resume_dir:
            # Batch mode
            resume_dir = Path(args.resume_dir)
            if not resume_dir.is_dir():
                logger.error("Resume directory not found: %s", resume_dir)
                sys.exit(1)

            resume_files = sorted(
                p for p in resume_dir.iterdir()
                if p.suffix.lower() in (".pdf", ".docx") and p.is_file()
            )

            if not resume_files:
                logger.error("No PDF/DOCX files found in: %s", resume_dir)
                sys.exit(1)

            logger.info("Found %d resume files", len(resume_files))
            reports = pipeline.process_batch(
                jd_path=args.jd,
                resume_paths=resume_files,
                output_dir=args.output,
            )
            print(f"\n✅ Generated {len(reports)} reports in: {args.output}")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        print(f"\n❌ Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
