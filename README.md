# Harnessing-GPT-3.5-for-Text-Parsing-in-Solid-State-Synthesis-case-study-of-ternary-chalchogenides

In solid-state thermoelectrics, single-phase compounds are necessary for expanding the field. Thermoelectric devices, which convert heat into electricity and vice-versa, rely on bulk, single phase materials, usually made using solid-state synthesis. However, traditional text-extraction algorithms require careful data curation from a large corpus of text to construct, which may not be available for some subfields. By leveraging on GPT-3.5, we were able to efficiently parse text to extract and infer synthesis conditions for ternary chalchogenides (A,B and X where A, B are common metallic elements, and X is a chalchogen)  from a database of over 100 research papers. We introduce a template for parsing solid-state synthesis recipes that encapsulates all the essential information for an end user, proposing an intuitive perspective of viewing these recipes in terms of primary and secondary heating peaks. We validate the inherent anthropogenic bias endemic in published papers – only positive results are reported. Our methodology not only demonstrates the efficiency of this automated approach but also underscores the generalizability of LLMs for such tasks. We believe our work provides a roadmap for future endeavors seeking to amalgamate LLMs with material science research, heralding a potentially transformative paradigm in the synthesis and characterization of novel materials.

We provide 3 datasets
1. gold_standard - manually extracted CuInTe/Se
2. silver_standard - GPT extracted from gold_standard
3. exchsp - GPT-extract ternary chalcogenides

To run the same prompts, download LLM_data_extraction_template.ipynb and utils.py, and follow the instructions in the notebook.
Results for ExChSp are placed seperately.

Additionally, the code for feature importance analysis and other figures are also provided in visualization.ipynb.
