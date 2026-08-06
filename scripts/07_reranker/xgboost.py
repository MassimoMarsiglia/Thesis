# import numpy as np
# from datasets import load_dataset
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     roc_auc_score,
# )
# from xgboost import XGBClassifier

# # ============================================================
# # CONFIG
# # ============================================================

# DATA_FILE = "data/train/mined_training_pairs_v2.jsonl"

# MODEL_NAME = "artifacts/models/embeddinggemma-document-linker/checkpoint-170"

# BATCH_SIZE = 32


# # ============================================================
# # LOAD DATA
# # ============================================================

# dataset = load_dataset("json", data_files=DATA_FILE)["train"]

# print("Examples:", len(dataset))
# print("Columns:", dataset.column_names)
# print(dataset[0])


# # ============================================================
# # CHECK LABEL DISTRIBUTION
# # ============================================================

# labels = dataset["label"]

# unique, counts = np.unique(labels, return_counts=True)

# print("Label distribution:", dict(zip(unique, counts)))

# # ============================================================
# # SBERT MODEL
# # ============================================================

# encoder = SentenceTransformer(MODEL_NAME)


# # ============================================================
# # FEATURE CREATION
# # ============================================================


# def make_features(q, d):
#     """
#     Combine query/document embeddings
#     into XGBoost features.
#     """

#     cosine = np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))

#     return np.concatenate(
#         [
#             q,
#             d,
#             q * d,
#             np.abs(q - d),
#             [cosine],
#         ]
#     )


# # ============================================================
# # EMBEDDING DATASET
# # ============================================================


# def build_features(data):

#     queries = data["sentence_1"]
#     documents = data["sentence_2"]
#     labels = np.array(data["label"])

#     print("Encoding:", len(labels), "pairs")

#     q_embeddings = encoder.encode(
#         queries,
#         batch_size=BATCH_SIZE,
#         show_progress_bar=True,
#         normalize_embeddings=True,
#     )

#     d_embeddings = encoder.encode(
#         documents,
#         batch_size=BATCH_SIZE,
#         show_progress_bar=True,
#         normalize_embeddings=True,
#     )

#     X = np.array(
#         [
#             make_features(q, d)
#             for q, d in zip(
#                 q_embeddings,
#                 d_embeddings,
#             )
#         ]
#     )

#     return X, labels


# # ============================================================
# # BUILD TRAINING DATA
# # ============================================================

# X_train, y_train = build_features(train_data)

# print(
#     "Train shape:",
#     X_train.shape,
# )


# # ============================================================
# # TRAIN XGBOOST
# # ============================================================

# model = XGBClassifier(
#     n_estimators=500,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="binary:logistic",
#     eval_metric="logloss",
#     tree_method="hist",
#     n_jobs=-1,
# )


# model.fit(
#     X_train,
#     y_train,
# )


# # ============================================================
# # EVALUATION
# # ============================================================

# X_test, y_test = build_features(test_data)


# probabilities = model.predict_proba(X_test)[:, 1]


# predictions = (probabilities >= 0.5).astype(int)


# accuracy = accuracy_score(y_test, predictions)


# auc = roc_auc_score(y_test, probabilities)


# precision, recall, f1, _ = precision_recall_fscore_support(
#     y_test,
#     predictions,
#     average="binary",
# )


# print("\nEvaluation")
# print("----------------")
# print(f"Accuracy : {accuracy:.4f}")
# print(f"ROC-AUC  : {auc:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1       : {f1:.4f}")


# # ============================================================
# # RERANKING
# # ============================================================


# def rerank(
#     query,
#     candidate_documents,
# ):

#     q_embedding = encoder.encode(
#         [query],
#         normalize_embeddings=True,
#     )[0]

#     doc_embeddings = encoder.encode(
#         candidate_documents,
#         batch_size=BATCH_SIZE,
#         normalize_embeddings=True,
#     )

#     X = np.array(
#         [
#             make_features(
#                 q_embedding,
#                 d_embedding,
#             )
#             for d_embedding in doc_embeddings
#         ]
#     )

#     scores = model.predict_proba(X)[:, 1]

#     ranked = sorted(
#         zip(candidate_documents, scores),
#         key=lambda x: x[1],
#         reverse=True,
#     )

#     return ranked

# example = test_data[0]


# query = example["sentence_1"]


# # Use real candidates:
# # one positive + some negatives

# candidates = [
#     example["sentence_2"],
#     # add more retrieved documents here
# ]


# results = rerank(
#     query,
#     candidates,
# )


# print("\nRanking")
# print("----------------")

# for doc, score in results:
#     print("Score:", round(float(score), 4))
#     print(doc[:300])
#     print("----------------")
