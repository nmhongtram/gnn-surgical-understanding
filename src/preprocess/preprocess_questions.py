import os
import h5py
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

import src.config as cfg


# Valid ana_type values to avoid confusion with coordinates or other attributes
VALID_ANA_TYPES = {
    'zero_hop.json',
    'one_hop.json', 
    'single_and.json'
}


def normalize_answer(answer):
    """
    Normalize answer labels to handle duplicates/variations
    
    Args:
        answer: Original answer string
        
    Returns:
        Normalized answer string
    """
    # Handle specimen bag variations
    if answer.strip().lower() == "specimenbag":
        return "specimen_bag"
    
    # Add other normalizations here if needed
    # Example: answer = answer.strip().lower()
    
    return answer.strip()


def load_questions_from_mode(mode, ana_type=None):
    """
    Load all questions from a specific mode (train/val/test/debug)
    
    Args:
        mode: Dataset split mode
        ana_type: List of analysis types to filter by (e.g., ['zero_hop.json', 'one_hop.json'])
                 If None, load all questions
    """
    questions = []
    qa_folder_path = cfg.QUESTIONS_DIR / mode
    file_list = list(qa_folder_path.glob("*.txt"))
    
    print(f"📁 Processing {mode} mode: {len(file_list)} files")
    if ana_type:
        print(f"🔍 Filtering by ana_type: {ana_type}")
    
    for file in tqdm(file_list, desc=f"Loading {mode} questions"):
        try:
            with open(file, "r", encoding='utf-8') as file_data:
                lines = [line.strip("\n") for line in file_data if line.strip() != ""]
                
                for idx, line in enumerate(lines):
                    if idx >= 2 and "|" in line:  # Skip header lines
                        parts = line.split("|")
                        if len(parts) >= 2:
                            question = parts[0].strip()
                            answer = normalize_answer(parts[1].strip())
                            
                            # Extract ana_type information if available
                            ana_type_info = None
                            query_type = None
                            location = None
                            
                            if len(parts) >= 3:
                                potential_ana_type = parts[2].strip()
                                # Only set ana_type if it's a valid value (avoid coordinates confusion)
                                if potential_ana_type in VALID_ANA_TYPES:
                                    ana_type_info = potential_ana_type
                                    
                            if len(parts) >= 4:
                                query_type = parts[3].strip()
                            if len(parts) >= 5:
                                location = parts[4].strip()
                            
                            # Filter by ana_type if specified
                            if ana_type:
                                if not ana_type_info or ana_type_info not in ana_type:
                                    continue  # Skip this question if no valid ana_type or not in filter
                            
                            # Store question with full metadata
                            questions.append({
                                'question': question,
                                'answer': answer,
                                'file': file.stem,  # Video frame ID  
                                'line_idx': idx,
                                'ana_type': ana_type_info,
                                'query_type': query_type,
                                'location': location
                            })
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
            continue
    
    print(f"✅ Loaded {len(questions)} questions from {mode}")
    
    # Print validation statistics
    total_with_valid_ana_type = sum(1 for q in questions if q.get('ana_type'))
    if total_with_valid_ana_type > 0:
        print(f"📋 {total_with_valid_ana_type} questions have valid ana_type")
    
    # Print ana_type statistics if filtering was applied
    if ana_type or any(q.get('ana_type') for q in questions):
        ana_type_counts = {}
        for q in questions:
            at = q.get('ana_type', 'unknown')
            ana_type_counts[at] = ana_type_counts.get(at, 0) + 1
        
        print("📊 Ana_type distribution:")
        # Sort with None-safe key
        for at, count in sorted(ana_type_counts.items(), key=lambda x: x[0] or 'zzz_none'):
            print(f"   {at or 'None'}: {count} questions")
    
    return questions


def tokenize_questions_batch(questions, tokenizer, batch_size=64, max_length=128):
    """
    Tokenize questions in batches for efficiency
    
    Args:
        questions: List of question dictionaries
        tokenizer: BERT tokenizer
        batch_size: Number of questions to process at once
        max_length: Maximum sequence length
        
    Returns:
        Dict with tokenized data
    """
    print(f"🔤 Tokenizing {len(questions)} questions...")
    
    # Extract just the question texts
    question_texts = [q['question'] for q in questions]
    
    # Store tokenized results
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    
    # Process in batches
    for i in tqdm(range(0, len(question_texts), batch_size), desc="Tokenizing batches"):
        batch_questions = question_texts[i:i+batch_size]
        
        # Tokenize batch
        tokenized = tokenizer(
            batch_questions,
            padding='max_length',  # Pad to max_length for consistent shape
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Store results
        all_input_ids.append(tokenized['input_ids'])
        all_attention_masks.append(tokenized['attention_mask'])
        
        # Some tokenizers don't return token_type_ids
        if 'token_type_ids' in tokenized:
            all_token_type_ids.append(tokenized['token_type_ids'])
    
    # Concatenate all batches
    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'questions': questions  # Keep original metadata
    }
    
    if all_token_type_ids:
        token_type_ids = torch.cat(all_token_type_ids, dim=0)
        result['token_type_ids'] = token_type_ids
    
    print(f"✅ Tokenized shape: {input_ids.shape}")
    return result


def save_tokenized_data(tokenized_data, output_path):
    """Save tokenized data to HDF5 file"""
    print(f"💾 Saving tokenized data to {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save tokenized tensors
        f.create_dataset('input_ids', data=tokenized_data['input_ids'].numpy(), compression='gzip')
        f.create_dataset('attention_mask', data=tokenized_data['attention_mask'].numpy(), compression='gzip')
        
        if 'token_type_ids' in tokenized_data:
            f.create_dataset('token_type_ids', data=tokenized_data['token_type_ids'].numpy(), compression='gzip')
        
        # Save metadata as JSON strings
        questions_json = [json.dumps(q) for q in tokenized_data['questions']]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('questions_metadata', data=questions_json, dtype=dt, compression='gzip')
        
        # Save tokenizer info
        f.attrs['num_questions'] = len(tokenized_data['questions'])
        f.attrs['max_length'] = tokenized_data['input_ids'].shape[1]
        f.attrs['vocab_size'] = tokenized_data['input_ids'].max().item() + 1
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved {len(tokenized_data['questions'])} tokenized questions ({file_size:.1f} MB)")


def main():
    """Main preprocessing function"""
    import argparse
    
    # Add command line arguments for ana_type filtering
    parser = argparse.ArgumentParser(description='Pre-tokenize questions with optional ana_type filtering')
    parser.add_argument('--ana_type', nargs='*', 
                       help='Filter by analysis types (e.g., --ana_type zero_hop.json one_hop.json)')
    parser.add_argument('--modes', nargs='*', default=['debug', 'train', 'val', 'test', 'test_full'],
                       help='Modes to process (default: all)')
    
    args = parser.parse_args()
    
    print("🚀 Starting question pre-tokenization...")
    
    if args.ana_type:
        print(f"🔍 Filtering by ana_type: {args.ana_type}")
    
    # Initialize tokenizer
    print("🔤 Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    
    # Process each mode
    modes = args.modes
    
    for mode in modes:
        print(f"\n📋 Processing {mode.upper()} mode...")
        
        try:
            # Check if mode directory exists
            mode_path = cfg.QUESTIONS_DIR / mode
            if not mode_path.exists():
                print(f"⚠️ Skipping {mode} - directory not found: {mode_path}")
                continue
            
            # Load questions with ana_type filtering
            questions = load_questions_from_mode(mode, ana_type=args.ana_type)
            
            if not questions:
                print(f"⚠️ No questions found in {mode} mode")
                continue
            
            # Tokenize questions
            tokenized_data = tokenize_questions_batch(
                questions, tokenizer, 
                batch_size=64,  # Adjust based on available memory
                max_length=128   # Sufficient for most surgical questions
            )
            
            # Save tokenized data with ana_type suffix if filtered
            if args.ana_type:
                ana_suffix = "_".join(args.ana_type).replace('.json', '')
                output_path = cfg.TOKENIZED_QUESTIONS_DIR / f"tokenized_{mode}_{ana_suffix}.h5"
            else:
                output_path = cfg.TOKENIZED_QUESTIONS_DIR / f"tokenized_{mode}.h5"
            save_tokenized_data(tokenized_data, output_path)
            
        except Exception as e:
            print(f"❌ Error processing {mode}: {e}")
            continue
    
    print("\n🎉 Question pre-tokenization completed!")
    
    # Print summary
    print("\n📊 Summary:")
    for mode in modes:
        # Check both filtered and unfiltered files
        tokenized_paths = list(cfg.QUESTIONS_DIR.glob(f"tokenized_{mode}*.h5"))
        for tokenized_path in tokenized_paths:
            if tokenized_path.exists():
                file_size = tokenized_path.stat().st_size / (1024 * 1024)
                print(f"   {tokenized_path.name}: ({file_size:.1f} MB)")
    
    print("\n💡 Usage Examples:")
    print("   # All questions:")
    print("   python preprocess_questions.py")
    print("   # Only zero-hop questions:")  
    print("   python preprocess_questions.py --ana_type zero_hop.json")
    print("   # Multiple ana_types:")
    print("   python preprocess_questions.py --ana_type zero_hop.json one_hop.json")
    print("   # Specific modes only:")
    print("   python preprocess_questions.py --modes train val")


if __name__ == "__main__":
    main()