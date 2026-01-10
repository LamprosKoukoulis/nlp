import matplotlib.pyplot as plt
from grammartree import num_of_gram_errors
import numpy as np
width =0.15

def graph_errors(models,paragraphs,result_text):
    x = np.arange(len(paragraphs))  # paragraph indices
    gram_errors_org =[]
    gram_errors_cor ={model: [] for model in models}

    for i,p in enumerate(paragraphs):
        er_org,er_org = num_of_gram_errors(p,p)
        gram_errors_org.append(er_org)

        print(f"Grammatical errors in Paragraph {i+1}:")

        for model_name in models:
            er_org, er_cor = num_of_gram_errors(p,result_text[model_name][i])
            gram_errors_cor[model_name].append(er_cor)

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x, gram_errors_org, width,color = 'gray', label="Original")
    for i, model_name in enumerate(models):
        ax.bar(x +(i+1) * width, gram_errors_cor[model_name], width, label=f"{model_name} Corrected")

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(paragraphs))])
    ax.set_ylabel("Number of Grammatical Errors")
    ax.set_title("Grammatical Errors per Paragraph and Model")
    ax.legend()
    plt.show()


def graph_sematic(models,paragraphs,sematic_results):
    x = np.arange(len(paragraphs))  # paragraph indices
    fig, ax = plt.subplots()
    
    for i, model_name in enumerate(models):
        scores = sematic_results[model_name]
        ax.bar(x + i*width, scores, width, label=model_name)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(paragraphs))])
    ax.set_ylim(0,1)
    ax.set_ylabel("Semantic Similarity")
    ax.set_title("Semantic Similarity per Paragraph and Model")
    ax.legend()
    plt.show()