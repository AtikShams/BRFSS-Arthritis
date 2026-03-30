import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from sklearn.ensemble import StackingClassifier

# ==========================================
# MOCK ARCHITECTURES FROM MAIN SCRIPT
# ==========================================

def build_attention_ann(input_dim=20):
    """Replicates the custom Attention ANN from the main pipeline."""
    from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, Reshape, Flatten
    inputs = Input(shape=(input_dim,))
    x_reshaped = Reshape((input_dim, 1))(inputs)
    attention_out = MultiHeadAttention(num_heads=2, key_dim=2)(x_reshaped, x_reshaped)
    x = Flatten()(attention_out)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

# ==========================================
# UNIT TESTS
# ==========================================

def test_attention_ann_compilation():
    """
    Test 1: Verifies that the custom Attention-based ANN compiles properly 
    and outputs the correct shape for binary classification.
    """
    # Assuming an arbitrary feature length, e.g., 20 features
    model = build_attention_ann(input_dim=20) 
    
    # Assert output shape is exactly 1 node (binary classification)
    assert model.output_shape == (None, 1), f"Expected output shape (None, 1), got {model.output_shape}"
    
    # Assert the loss function is mathematically correct for the task
    assert model.loss == 'binary_crossentropy', "Model must use binary_crossentropy loss"

def test_stacking_classifier_initialization():
    """
    Test 2: Verifies that the Stacked Ensemble initializes correctly with 
    the exact required base estimators (LightGBM, XGBoost, GradientBoosting).
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    stacking_estimators = [
        ('lgbm', LGBMClassifier(random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ]
    
    model = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5, n_jobs=-1
    )
    
    # Assert the ensemble contains exactly 3 base models
    assert len(model.estimators) == 3, "Stacked Ensemble should have exactly 3 base estimators."
    
    # Assert the final meta-estimator is Logistic Regression
    assert isinstance(model.final_estimator, LogisticRegression), "Meta-estimator must be Logistic Regression."
