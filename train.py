import sys
import traceback
import logging

from finetrainers import (BaseArgs, ControlTrainer, ReferenceTrainer,
                          SFTTrainer, TrainingType, get_logger)
from finetrainers.config import _get_model_specifiction_cls
from finetrainers.trainer.control_trainer.config import (ControlFullRankConfig,
                                                         ControlLowRankConfig)
from finetrainers.trainer.reference_trainer.config import ReferenceConfig
from finetrainers.trainer.sft_trainer.config import (SFTFullRankConfig,
                                                     SFTLowRankConfig)

# Prevent duplicate logs by ensuring the root logger doesn't propagate to parent
logging.getLogger('finetrainers').propagate = False

logger = get_logger()


def main():
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            f'Failed to set multiprocessing start method to "fork". This can lead to poor performance, high memory usage, or crashes. '
            f"See: https://pytorch.org/docs/stable/notes/multiprocessing.html\n"
            f"Error: {e}"
        )

    try:
        args = BaseArgs()

        argv = [y.strip() for x in sys.argv for y in x.split()]
        training_type_index = argv.index("--training_type")
        if training_type_index == -1:
            raise ValueError("Training type not provided in command line arguments.")

        training_type = argv[training_type_index + 1]
        training_cls = None
        if training_type == TrainingType.LORA:
            training_cls = SFTLowRankConfig
        elif training_type == TrainingType.FULL_FINETUNE:
            training_cls = SFTFullRankConfig
        elif training_type == TrainingType.CONTROL_LORA:
            training_cls = ControlLowRankConfig
        elif training_type == TrainingType.CONTROL_FULL_FINETUNE:
            training_cls = ControlFullRankConfig
        elif training_type in [TrainingType.REFERENCE_LORA, TrainingType.REFERENCE_FULL_FINETUNE]:
            training_cls = ReferenceConfig
        else:
            raise ValueError(f"Training type {training_type} not supported.")

        # For reference trainer, load from JSON if dataset_config is provided
        dataset_config = None
        for i, arg in enumerate(argv):
            if arg == "--dataset_config" and i + 1 < len(argv):
                dataset_config = argv[i + 1]
                break
            elif arg.startswith("--dataset_config="):
                dataset_config = arg.split("=", 1)[1]
                break
                
        if training_type in [TrainingType.REFERENCE_LORA, TrainingType.REFERENCE_FULL_FINETUNE] and dataset_config:
            logger.info(f"Loading reference configuration from JSON: {dataset_config}")
            config = ReferenceConfig.from_json(dataset_config)
            args.register_args(config)
        else:
            # Use default configuration for non-reference trainers or when no dataset_config
            args.register_args(training_cls())
        args = args.parse_args()

        model_specification_cls = _get_model_specifiction_cls(args.model_name, args.training_type)
        model_specification = model_specification_cls(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            tokenizer_id=args.tokenizer_id,
            tokenizer_2_id=args.tokenizer_2_id,
            tokenizer_3_id=args.tokenizer_3_id,
            text_encoder_id=args.text_encoder_id,
            text_encoder_2_id=args.text_encoder_2_id,
            text_encoder_3_id=args.text_encoder_3_id,
            transformer_id=args.transformer_id,
            vae_id=args.vae_id,
            image_encoder_id=getattr(args, 'image_encoder_id', None),
            image_processor_id=getattr(args, 'image_processor_id', None),
            text_encoder_dtype=args.text_encoder_dtype,
            text_encoder_2_dtype=args.text_encoder_2_dtype,
            text_encoder_3_dtype=args.text_encoder_3_dtype,
            transformer_dtype=args.transformer_dtype,
            vae_dtype=args.vae_dtype,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )

        if args.training_type in [TrainingType.LORA, TrainingType.FULL_FINETUNE]:
            trainer = SFTTrainer(args, model_specification)
        elif args.training_type in [TrainingType.CONTROL_LORA, TrainingType.CONTROL_FULL_FINETUNE]:
            trainer = ControlTrainer(args, model_specification)
        elif args.training_type in [TrainingType.REFERENCE_LORA, TrainingType.REFERENCE_FULL_FINETUNE]:
            trainer = ReferenceTrainer(args, model_specification)
        else:
            raise ValueError(f"Training type {args.training_type} not supported.")

        trainer.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()