from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import language_tool_python
import warnings

warnings.filterwarnings('ignore')
def create_grammar_tree(text:str)-> None:
    tool = language_tool_python.LanguageTool("en-US",config={"forceEncoding": "UTF-8"})

    matches = tool.check(text)
    
    root = Node(f"Text: {text}")
    #print(root)
    
    #  sentence node
    sentence_node = Node("Sentence Analysis", parent=root)
    #print(sentence_node)
    #  grammar issues node
    if matches:
        issues_node = Node("Grammar Issues", parent=sentence_node)
        for match in matches:
            #  issue node
            issue_node = Node(f"Issue: {match.ruleId}", parent=issues_node)
            Node(f"Message: {match.message}", parent=issue_node)
            Node(f"Context: {match.context}", parent=issue_node)
            if (match.replacements):
                suggestions_node = Node("Suggestions", parent=issue_node)
                for replacement in match.replacements:
                    Node(replacement,parent=suggestions_node)
    else:
        Node("No grammar issues found", parent=sentence_node)
    
    print("Grammar Analysis Tree:")
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre,node.name))
    
    # Export tree as PNG (requires graphviz)
    try:
        DotExporter(root).to_picture("grammar_tree.png")
        print("\nTree visualization saved as 'grammar_tree.png'")
    except Exception as e:
        print("\nCould not create PNG visualization. Make sure graphviz is installed.")
    tool.close()
    #return corrected_text, matches



def check_grammar(text:str):
    tool = language_tool_python.LanguageTool("en-US")
    is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and rule.replacements[0][0].isupper()
    corrected_text = text
    count =0
    while True:
        count+=1
        matches= tool.check(corrected_text)
        matches = [rule for rule in matches if not is_bad_rule(rule)]

        if not matches:
            break

        for match in sorted(matches,key=lambda m: m.offset,reverse=True):
            if match.replacements:
                replacement = match.replacements[0]
                start,end= match.offset, match.offset + match.errorLength
                corrected_text = corrected_text[:start] + replacement + corrected_text[end:]
    tool.close()
    #print("Original text:", text)
    print("Corrected text:\n", corrected_text)
    print("Number of iterations:",count)
    #print("Grammar issues found:", len(matches))

    return matches


def num_of_gram_errors(original:str,corrected:str)-> tuple[int,int]:
    tool = language_tool_python.LanguageTool("en-US")
    errors_original = len(tool.check(original))
    errors_corrected = len(tool.check(corrected))
    print(f"In original Text: {errors_original}\tIn Corrected Text: {errors_corrected}")
    tool.close()
    return errors_original, errors_corrected


