import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from data_loader import load_pavia_data

def prepare_tabular_data(data: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flattens the 3D hyperspectral cube and removes unclassified background pixels."""
    bands = data.shape[-1]
    
    X = data.reshape(-1, bands)
    y = gt.reshape(-1)
    
    valid_pixels = y > 0
    
    X_clean = X[valid_pixels]
    y_clean = y[valid_pixels] - 1
    
    return X_clean, y_clean

def evaluate_baseline(X: np.ndarray, y: np.ndarray, n_components: int = 100) -> None:
    """Trains and evaluates a PCA + Random Forest pipeline."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    target_names = [
        "Asphalt", "Meadows", "Gravel", "Trees", "Metal Sheets", 
        "Bare Soil", "Bitumen", "Bricks", "Shadows"
    ]
    
    print("\n--- Baseline Model Performance ---")
    print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == "__main__":
    data, gt = load_pavia_data()
    
    X, y = prepare_tabular_data(data, gt)
    
    evaluate_baseline(X, y)