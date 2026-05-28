import joblib

model = joblib.load("major_category_model_for_task2.pkl")
print("Model Pipeline:")
print(model)
print("\nSteps:")
for name, step in model.steps:
    print(f"  Step '{name}': {type(step)}")
    if name == 'preprocessor':
        print("  Preprocessor transformers:")
        for trans_name, trans_obj, trans_cols in step.transformers_:
            print(f"    - '{trans_name}': {type(trans_obj)} on {trans_cols}")
