from langchain.schema import HumanMessage
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_qa_pairs(df, critique_prompts, chat_model):
    """Evaluate QA pairs using the specified criteria."""
    # Create a fresh copy to avoid reference issues
    result_df = df.copy(deep=True)
    
    # Create new columns for the evaluations
    for crit in critique_prompts.keys():
        result_df[f"{crit}_score"] = None
        result_df[f"{crit}_eval"] = None
    
    print("Generating critique for each QA pair...")
    
    # iterate through each row, extract the question and context, feed the LLM with the prompt and question/context
    for idx, row in tqdm(result_df.iterrows(), total=len(result_df)):
        question = row["input"]
        
        # Prepare context
        context_val = row["context"]
        if isinstance(context_val, list):
            context_str = " ".join(context_val)
        else:
            context_str = context_val
        
        # Evaluate each criterion
        for criterion, prompt_template in critique_prompts.items():
            if criterion == "groundedness":
                prompt = prompt_template.format(question=question, context=context_str)
            else:
                prompt = prompt_template.format(question=question)
            
            evaluation = chat_model.predict_messages([HumanMessage(content=prompt)])
            score, eval_text = extract_score_and_eval(evaluation.content)
            
            result_df.at[idx, f"{criterion}_score"] = score
            result_df.at[idx, f"{criterion}_eval"] = eval_text
    
    print("Critique generation completed.")
    return result_df

def extract_score_and_eval(evaluation_text):
    """Extract score and evaluation text from model response."""
    if not evaluation_text:
        return None, None
    
    try:
        parts = evaluation_text.split("Total rating:")
        total_rating_part = parts[-1].strip()
        score = int(total_rating_part.split()[0])
        
        eval_text = evaluation_text.split("Total rating:")[0].split("Evaluation:")[-1].strip()
        
        return score, eval_text
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None, None

def process_evaluation_scores(
    df: pd.DataFrame,
    groundedness_weight: float = 0.35,
    standalone_weight: float = 0.35,
    relevance_weight: float = 0.3,
    bottom_percent_remove: float = 10
) -> pd.DataFrame:
    """
    Process evaluation scores by normalizing, applying transformations and filtering.
    """
    # Create a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    
    # Normalize scores to [0,1] range (original scores are in [1,5])
    processed_df["grounded_norm"] = (processed_df["groundedness_score"] - 1) / 4
    processed_df["standalone_norm"] = (processed_df["standalone_score"] - 1) / 4
    processed_df["relevance_norm"] = (processed_df["relevance_score"] - 1) / 4
    
    # Apply mid-range preference transformation to groundedness
    # This transforms the score so that values around 0.5 get higher scores
    processed_df["grounded_norm"] = 1 - abs(processed_df["grounded_norm"] - 0.5) * 1.5
    
    # Calculate weighted total score
    processed_df["total_score"] = (
        groundedness_weight * processed_df["grounded_norm"] +
        standalone_weight * processed_df["standalone_norm"] +
        relevance_weight * processed_df["relevance_norm"]
    )
    
    # Remove bottom N% based on total score
    threshold = np.percentile(processed_df["total_score"], bottom_percent_remove)
    processed_df = processed_df[processed_df["total_score"] > threshold].reset_index(drop=True)
    
    return processed_df

def analyze_semantic_similarity(
    df,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    Compute and visualise the pair-wise semantic similarity of the generated questions.
    For typical RAG test sets, a mean below ~0.40 is often considered “diverse enough,” while means above ~0.60 suggest many near-duplicates.
    """
    # 1) load model and create embeddings
    model = SentenceTransformer(model_name)
    texts = df["input"].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 2) calculate cosine sim
    sim_matrix = cosine_similarity(embeddings)
    
    # 3) extract upper triangle of the matrix
    i_upper, j_upper = np.triu_indices_from(sim_matrix, k=1)
    sims = sim_matrix[i_upper, j_upper]
    
    # 4) get stats
    mean_sim = sims.mean()
    std_sim = sims.std()
    print(f"Mean similarity: {mean_sim:.4f}")
    print(f"Std  similarity: {std_sim:.4f}")
    
    # 5) plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(sims, bins=30, alpha=0.7)
    plt.axvline(mean_sim, color='r', linestyle='--',
                label=f"Mean: {mean_sim:.3f}")
    plt.title("Distribution of Query Similarities")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    
    return sims, sim_matrix