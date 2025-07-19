from datasets import load_dataset

def check_dataset():
    """
    Check the actual structure of the training dataset
    """
    print("[INFO] Loading dataset to inspect structure...")
    dataset = load_dataset("elenigkove/Email_Intent_Classification")
    
    print(f"[INFO] Dataset keys: {dataset.keys()}")
    print(f"[INFO] Train samples: {len(dataset['train'])}")
    print(f"[INFO] Test samples: {len(dataset['test'])}")
    
    print("\n[INFO] Dataset features:")
    print(dataset['train'].features)
    
    print("\n[INFO] Sample data (first 3 examples):")
    for i in range(3):
        example = dataset['train'][i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Email: {example['Email'][:100]}...")
        print(f"Intent: {example['Intent']}")
    
    print("\n[INFO] All unique intent categories:")
    all_intents = set(dataset['train']['Intent']) 
    print(sorted(list(all_intents)))

if __name__ == "__main__":
    check_dataset()
