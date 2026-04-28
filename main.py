import os
from generate_synthetic_data import generate_synthetic_data
from train import ModelTrainer

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    generate_synthetic_data(num_samples=1000)
    trainer = ModelTrainer()
    trainer.train_model()
    print("✅ System Ready. Run 'streamlit run app.py'")