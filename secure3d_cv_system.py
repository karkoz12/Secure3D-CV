import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from typing import Tuple, Any

# --- 1. Secure3D-CV Core Logic (Section 2.5) ---

class Secure3DCV:
    """
    Implements the Secure3D-CV hybrid risk synthesis system.
    Combines a Statistical Anomaly Detector (A(S)) and a Learned Classifier (f_theta(x)).
    """
    def __init__(self):
        # Statistical Anomaly Detector (A(S))
        # IsolationForest is used as a proxy for the statistical anomaly score
        self.anomaly_model = IsolationForest(contamination='auto', random_state=42)
        self.anomaly_scaler = MinMaxScaler()
        
        # Learned Classifier (f_theta(x))
        self.classifier_model = LogisticRegression(solver='liblinear', random_state=42)
        
        # Optimal alpha value found during sensitivity analysis (Section 4.2)
        self.optimal_alpha = 0.6 

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains both the anomaly detector (on benign data) and the classifier (on all data).
        """
        # 1. Train Anomaly Detector (A(S)) on Benign Data only
        X_benign = X_train[y_train == 0]
        self.anomaly_model.fit(X_benign)
        
        # Normalize the anomaly scores (A(S)) to [0, 1] range
        # We fit the scaler on the decision function output of the benign data
        anomaly_scores_benign = self.anomaly_model.decision_function(X_benign)
        self.anomaly_scaler.fit(anomaly_scores_benign.reshape(-1, 1))
        
        # 2. Train Learned Classifier (f_theta(x)) on all data
        self.classifier_model.fit(X_train, y_train)

    def _calculate_anomaly_score_norm(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates and normalizes the statistical anomaly score A(S) to [0, 1].
        """
        # IsolationForest decision_function output is negative for inliers, positive for outliers
        # We invert and scale it so that higher values mean higher anomaly (closer to 1)
        raw_scores = self.anomaly_model.decision_function(X)
        
        # Scale the raw scores using the scaler fitted on benign data
        scaled_scores = self.anomaly_scaler.transform(raw_scores.reshape(-1, 1))
        
        # Invert the scale: 1 - scaled_scores (so that 1 is high anomaly)
        # Note: The scaling is complex in practice, here we use a simple inversion
        # to ensure higher score means higher risk.
        A_norm = 1 - scaled_scores.flatten()
        
        # Clip to [0, 1] range
        A_norm = np.clip(A_norm, 0, 1)
        
        return A_norm

    def calculate_risk_score(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """
        Calculates the Hybrid Risk Score R(S) = alpha * A(S) + (1-alpha) * (1 - f_theta(x)).
        """
        if not hasattr(self, 'anomaly_model') or not hasattr(self, 'classifier_model'):
            raise RuntimeError("Model must be trained before calculating risk score.")

        # 1. Calculate A(S) (Normalized Anomaly Score)
        A_norm = self._calculate_anomaly_score_norm(X)
        
        # 2. Calculate (1 - f_theta(x)) (Probability of being UNSAFE)
        # f_theta(x) is the probability of being SAFE (Label 0)
        # We use predict_proba and take the probability of Label 1 (Attacked/Unsafe)
        # If the model is not confident, it returns 0.5 for both classes.
        
        # Get probability of being Label 1 (Unsafe)
        P_unsafe = self.classifier_model.predict_proba(X)[:, 1] 
        
        # 3. Hybrid Risk Synthesis R(S)
        R_score = alpha * A_norm + (1 - alpha) * P_unsafe
        
        return R_score

    def predict(self, X: np.ndarray, alpha: float = None, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts the label (0=Safe, 1=Unsafe) based on the risk score R(S).
        """
        if alpha is None:
            alpha = self.optimal_alpha
            
        R_score = self.calculate_risk_score(X, alpha)
        
        # Predict 1 (Unsafe) if R_score > threshold
        predictions = (R_score > threshold).astype(int)
        return predictions

# --- 2. Evaluation Function ---

def evaluate_secure3d_cv(X_test: np.ndarray, y_test: np.ndarray, model: Secure3DCV, alpha: float) -> Dict[str, float]:
    """
    Evaluates the Secure3D-CV model for a given alpha value.
    """
    predictions = model.predict(X_test, alpha=alpha)
    
    # Calculate metrics
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, model.calculate_risk_score(X_test, alpha=alpha))
    
    # Calculate TPR and FPR
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "alpha": alpha,
        "F1_Score": f1,
        "AUC": auc,
        "TPR": tpr * 100,
        "FPR": fpr * 100
    }

# --- 3. Main Execution Block (Integration with Data Generator) ---

if __name__ == '__main__':
    # Import the data generator functions
    try:
        from secure3d_cv_data_generator import generate_dataset
    except ImportError:
        print("Error: secure3d_cv_data_generator.py not found. Please ensure it is in the same directory.")
        exit()

    # --- Configuration ---
    N_SAMPLES = 2000
    ATTACK_RATIO = 0.2
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    print("--- Secure3D-CV System Simulation ---")
    
    # 1. Data Generation
    X_features, y_labels, _ = generate_dataset(n_samples=N_SAMPLES, attack_ratio=ATTACK_RATIO)
    
    # 2. Data Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_labels
    )
    
    print(f"\nTraining on {len(y_train)} samples, Testing on {len(y_test)} samples.")
    
    # 3. Model Training
    model = Secure3DCV()
    model.train(X_train, y_train)
    print("Secure3D-CV Model Trained (Anomaly Detector and Classifier).")
    
    # 4. Evaluation at Optimal Alpha (0.6)
    optimal_alpha = model.optimal_alpha
    results = evaluate_secure3d_cv(X_test, y_test, model, optimal_alpha)
    
    print(f"\n--- Results at Optimal Alpha ({optimal_alpha}) ---")
    print(f"F1 Score: {results['F1_Score']:.4f}")
    print(f"AUC Score: {results['AUC']:.4f}")
    print(f"True Positive Rate (TPR): {results['TPR']:.2f}%")
    print(f"False Positive Rate (FPR): {results['FPR']:.2f}%")
    
    # 5. Sensitivity Analysis (Data for Figure 2)
    print("\n--- Sensitivity Analysis (Data for Figure 2) ---")
    alpha_values = np.linspace(0.0, 1.0, 11)
    sensitivity_data = []
    
    for alpha in alpha_values:
        res = evaluate_secure3d_cv(X_test, y_test, model, alpha)
        sensitivity_data.append({
            "Alpha": f"{alpha:.1f}",
            "F1_Score": f"{res['F1_Score']:.4f}",
            "TPR": f"{res['TPR']:.2f}%",
            "FPR": f"{res['FPR']:.2f}%"
        })
        
    df_sensitivity = pd.DataFrame(sensitivity_data)
    print(df_sensitivity.to_markdown(index=False))
    
    # Note: The results here will not exactly match the paper's results (99.3% TPR)
    # because the data generation is a simplified simulation. However, the trend 
    # (peak F1-Score around alpha=0.6) should be visible.
```
