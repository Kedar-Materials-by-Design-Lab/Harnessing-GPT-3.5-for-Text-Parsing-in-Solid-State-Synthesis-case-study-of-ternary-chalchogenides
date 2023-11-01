# Extraction campaign for ExChSp dataset

These is where we parsed PDFs with solid-state synthesis of ternary chalcogenides asides from CIT/CIS. Note that there were a total of 168 papers that were manually downloaded from various sources, however, only 61 papers were successfully extracted and parsed. The rest of the papers were filtered by the initial subprompts and deemed to not include relevant information. 

Notably, there were a lot of time-out errors from the OpenAI server side. Due to this issue, we performed multiple extractions in parallel by manually running seperate Jupyter notebooks. We provide 4 examples (both successful single runs, and unsuccessful interrupted runs) to show you what it looks like. Do note that when the notebook stops printing text, it means that it is stuck in the timeout, and you will need to wait 10-15 minutes until the timeout error is returned. Afterwards you will need to reinitiate the loop from where you left off.


