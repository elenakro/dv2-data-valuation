import pickle
from src.evaluation import noisy_label_detection, point_removal

import warnings
import os
os.environ["PYTHONWARNINGS"] = "ignore"  # also for pyDVL's worker process
warnings.filterwarnings("ignore")

filename="experiments/banzhaf_results/banzhaf_500_5000_0.2_2dplanes.pkl"

def banzhaf_noisy_label_detection(filename):
    with open(filename, "rb") as f:
        saved_data = pickle.load(f)

        result       = saved_data["result"]
        elapsed_time = saved_data["execution_time_seconds"]
        
        values  = result.values
        indices = saved_data["noisy_idx"]

        precision, recall, f1 = noisy_label_detection(values, indices)
        print(f"Precission: {precision:.2f}")
        print(f"Recall:     {recall:.2f}")
        print(f"f1 Accuracy:{f1:.2f}")

def banzhaf_point_removal(filename):
     with open(filename, "rb") as f:
            saved_data = pickle.load(f)
    
            result  = saved_data["result"]
            X_train = saved_data["x_train"]
            y_train = saved_data["y_train"]
            X_test  = saved_data["x_test"]
            y_test  = saved_data["y_test"]
            elapsed_time = saved_data["execution_time_seconds"]
            
            values  = result.values
    
            accs = point_removal(values, X_train, y_train, X_test, y_test)

            print(accs)
            print(saved_data["peak_mem"])
            print(saved_data["baseline_mem"])
            print(saved_data["fit_count"])


if __name__ == "__main__": 
    banzhaf_noisy_label_detection(filename)
    banzhaf_point_removal(filename)



