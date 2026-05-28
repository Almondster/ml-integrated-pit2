import sys
import pickle
import joblib

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except AttributeError as e:
            print(f"FAILED to find class: '{name}' in module: '{module}'")
            raise e

print("Attempting to unpickle with detailed tracking...")
try:
    with open("major_category_model_for_task2.pkl", "rb") as f:
        # joblib numpy pickle uses custom loading but wraps pickle under the hood.
        # Let's try loading it via custom unpickler first to see the streams
        obj = joblib.load("major_category_model_for_task2.pkl")
        print("Success! Loaded successfully.")
except Exception as e:
    print(f"\nCaught exception: {type(e).__name__}: {e}")
