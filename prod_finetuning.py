# backend/app/services/finetune/dataset_builder.py
from typing import List, Dict
from datasets import Dataset
import json
import logging

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """Build training datasets from user feedback"""
    
    @staticmethod
    def build_from_queries(
        queries: List[Dict],
        min_rating: int = 4
    ) -> Dataset:
        """
        Build dataset from rated queries
        
        Args:
            queries: List of {query, answer, rating, sources}
            min_rating: Minimum rating to include
        
        Returns:
            HuggingFace Dataset
        """
        training_examples = []
        
        for q in queries:
            if q.get('rating', 0) < min_rating:
                continue
            
            # Format as instruction-following
            sources_text = "\n\n".join([
                f"[Source: {s['source']}]\n{s['text']}"
                for s in q.get('sources', [])
            ])
            
            instruction = f"""Answer the question using only the provided context.

Context:
{sources_text}

Question: {q['query']}"""
            
            training_examples.append({
                'instruction': instruction,
                'output': q['answer']
            })
        
        logger.info(f"Built dataset with {len(training_examples)} examples")
        return Dataset.from_list(training_examples)
    
    @staticmethod
    def build_from_corrections(
        corrections: List[Dict]
    ) -> Dataset:
        """
        Build dataset from manual corrections
        
        Args:
            corrections: List of {query, wrong_answer, correct_answer, sources}
        """
        examples = []
        
        for c in corrections:
            sources_text = "\n\n".join([
                f"[Source: {s['source']}]\n{s['text']}"
                for s in c.get('sources', [])
            ])
            
            examples.append({
                'instruction': f"Context:\n{sources_text}\n\nQuestion: {c['query']}",
                'output': c['correct_answer']
            })
        
        return Dataset.from_list(examples)

# backend/app/services/finetune/trainer.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import logging
import os

logger = logging.getLogger(__name__)

class LoRATrainer:
    """LoRA fine-tuning for Mistral"""
    
    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        output_dir: str = "data/finetune/checkpoints"
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ) -> str:
        """
        Fine-tune with LoRA
        
        Args:
            dataset: Training dataset
            num_epochs: Training epochs
            batch_size: Batch size (reduce if OOM)
            learning_rate: Learning rate
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: Dropout rate
        
        Returns:
            Path to saved adapter
        """
        logger.info("Loading base model...")
        
        # Load model in 4-bit for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Format as instruction-following
            texts = [
                f"[INST] {inst} [/INST] {out}"
                for inst, out in zip(examples['instruction'], examples['output'])
            ]
            return tokenizer(texts, truncation=True, max_length=2048, padding="max_length")
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            warmup_steps=100,
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save adapter
        adapter_path = os.path.join(self.output_dir, "final_adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        
        logger.info(f"Training complete. Adapter saved to: {adapter_path}")
        return adapter_path

# backend/app/services/finetune/evaluator.py
from typing import List, Dict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

class FineTuneEvaluator:
    """Evaluate fine-tuned model"""
    
    @staticmethod
    def evaluate(
        base_model_name: str,
        adapter_path: str,
        test_dataset: List[Dict]
    ) -> Dict:
        """
        Evaluate fine-tuned model vs base model
        
        Returns:
            {
                'base_model_perplexity': float,
                'finetuned_perplexity': float,
                'accuracy_improvement': float
            }
        """
        # Load models
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Evaluate
        base_perplexity = FineTuneEvaluator._calculate_perplexity(
            base_model, tokenizer, test_dataset
        )
        
        finetuned_perplexity = FineTuneEvaluator._calculate_perplexity(
            finetuned_model, tokenizer, test_dataset
        )
        
        improvement = ((base_perplexity - finetuned_perplexity) / base_perplexity) * 100
        
        return {
            'base_model_perplexity': base_perplexity,
            'finetuned_perplexity': finetuned_perplexity,
            'improvement_percent': round(improvement, 2)
        }
    
    @staticmethod
    def _calculate_perplexity(model, tokenizer, dataset):
        """Calculate perplexity on dataset"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for example in dataset:
                text = f"[INST] {example['instruction']} [/INST] {example['output']}"
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(dataset)
        perplexity = np.exp(avg_loss)
        
        return perplexity

# backend/app/tasks/training_tasks.py
from celery import shared_task
from app.services.finetune.trainer import LoRATrainer
from app.services.finetune.dataset_builder import DatasetBuilder
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def train_lora_adapter(
    self,
    queries: List[Dict],
    num_epochs: int = 3,
    min_rating: int = 4
):
    """
    Celery task for asynchronous training
    
    Usage:
        train_lora_adapter.delay(queries=[...], num_epochs=3)
    """
    try:
        logger.info(f"Building dataset from {len(queries)} queries...")
        dataset = DatasetBuilder.build_from_queries(queries, min_rating=min_rating)
        
        if len(dataset) < 100:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least 100 examples, got {len(dataset)}'
            }
        
        logger.info("Starting LoRA training...")
        trainer = LoRATrainer()
        adapter_path = trainer.train(dataset, num_epochs=num_epochs)
        
        return {
            'status': 'success',
            'adapter_path': adapter_path,
            'num_examples': len(dataset)
        }
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e)
        }