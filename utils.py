import time
import collections
import pandas as pd
from langchain.document_loaders import PyPDFLoader
import openai
import os

def extract_chunk(document,template_prompt, key):
    
    openai.api_key = key
    time.sleep(1)
    
    document = document.replace("  ", " ").replace("\n", "; ").replace(';',' ')

    prompt=template_prompt.replace('<document>',document)
    
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    tmp = response.choices[0].message["content"]
    print(tmp)
    
    return tmp

# Example prompt 1- 
document = '<document>'
template_prompt_1=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"Not specified\".

Example 1
Document: Polycrystalline samples of the composition Cu1-xNixInTe2 with nominal x values between 0 and 0.05 were prepared utilizing solid-state reactions. Polycrystalline samples were synthesized from a mixture of pure elements and compounds, Cu (4 N shots), NiTe2 (synthesized), In (5 N ingot) and Te (5 N chunks). The synthesis of NiTe2 was carried out by heating stoichiometric mixtures of 5 N Ni and Te to 1273 K for 2 hours in evacuated quartz ampoules. This material was powdered and mixed with Cu, In and Te in the ratio corresponding to the stoichiometry Cu1-xNixInTe2 (x = 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 and 0.05). The synthesis of the polycrystalline products were carried out in evacuated (10^-3 Pa), sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, maintained at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 823 K and 70 MPa for 1 h. Compacted disc-shaped samples (diameter 12 mm and thickness ca 2 mm) reached ≥95% of theoretical (X-ray) densities of the prepared compounds. To reach the homogenous distribution of the dopant and to stabilize the physical properties, the hot-pressed samples were annealed in a sealed quartz ampule under an argon atmosphere at T=773 K for 5 days.
Question: Does it include description of synthesis information? 
Answer: YES 

Example 2
Document: High-purity single elements Sn, Cu, In, and Te were weighed according to the nominal compositions of SnTe, CuInTe2 and CuSnmInTemþ2 (m ¼ 1, 3, 5, 7, 10, 14, 18), and then put inside 13 mm diameter fused quartz tubes. The tubes were sealed under vacuum (~104 torr) and slowly heated to 723 K in 12 h, then to 1423 K in 6 h, soaked at this temperature for 6 h and subsequently furnace cooling to room temperature. The resultant ingots were crushed  into fine powders and then densified by spark plasma sintering (SPS) method (SPS-211LX, Fuji Electronic Industrial Co., Ltd.) at 923 K for 5 min in a 12.7 mm diameter graphite die under an axial compressive stress of 40 MPa in vacuum. Highly dense (>96% of theoretical density) disk-shaped pellets with dimensions of 12.7 mm in diameter and 9 mm in thickness were obtained.
Question: Does it include description of synthesis information?  
Answer: YES 

Example 3
Document: SnTe, a lead-free analogue of PbTe, exhibits inferior thermoelectric performance due to its intrinsically high carrier concentration, and thus too low Seebeck coefficient and high thermal conductivity. To optimize the overall properties of SnTe, we composited SnTe with the other compound with opposite performance. CuInTe2 is a good choice due to its low carrier concentration, high Seebeck coefficient and relatively low thermal conductivity. As a result, the most optimized properties can be obtained in CuSnmInTemþ2 when m ¼ 7, and an enhanced figure of merit ZT can be obtained, ~1.0 at 873 K, which is a three-fold increase for pure CuInTe2 and one-fold increase for pure SnTe. The overall properties are optimized much greater than the expectation, which was estimated through Effective Medium Theory (EMT). Such deviation can be ascribed to the existence of high density of coherent nanostructures.
Question: Does it include description of synthesis information? 
Answer: No 

Document: \"\"\"{document}\"\"\"\nQuestion: Does it include description of synthesis information?  \nAnswer: '''
# Example prompt 2- 
template_prompt_2=f'''Answer all the following questions and give answer summary in list using the document provided in three \".
Answers should be in \"YES\" or \"NO\". If a particular piece of information is not present, output \"NA\".

Document: Polycrystalline samples of the composition Cu1-xNixInTe2 with nominal x values between 0 and 0.05 were prepared utilizing solid-state reactions. Polycrystalline samples were synthesized from a mixture of pure elements and compounds, Cu (4 N shots), NiTe2 (synthesized), In (5 N ingot) and Te (5 N chunks). The synthesis of NiTe2 was carried out by heating stoichiometric mixtures of 5 N Ni and Te to 1273 K for 2 hours in evacuated quartz ampoules. This material was powdered and mixed with Cu, In and Te in the ratio corresponding to the stoichiometry Cu1-xNixInTe2 (x = 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 and 0.05). The synthesis of the polycrystalline products were carried out in evacuated (10^-3 Pa), sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, maintained at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 823 K and 70 MPa for 1 h. Compacted disc-shaped samples (diameter 12 mm and thickness ca 2 mm) reached ≥95% of theoretical (X-ray) densities of the prepared compounds. To reach the homogenous distribution of the dopant and to stabilize the physical properties, the hot-pressed samples were annealed in a sealed quartz ampule under an argon atmosphere at T=773 K for 5 days.
1.Does it include synthesis information? 
2.Does the experiment resulted in pure phase formation of crystal? 
3.Is the experiment solid state synthesis?
Answer list: ['YES', 'NO', 'YES']

Document: \"\"\"{document}\"\"\"\n1.Does it include synthesis information? 
2.Does the experiment resulted in pure phase formation of crystal? 
3.Is the experiment solid state synthesis?
Answer list:'''
# Example prompt 3- 
template_prompt_3=f'''Answer the following question using the document provided in three \".
Answers should be in \"YES\" or \"NO\". If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition Cu1-xNixInTe2 with nominal x values between 0 and 0.05 were prepared utilizing solid-state reactions. Polycrystalline samples were synthesized from a mixture of pure elements and compounds, Cu (4 N shots), NiTe2 (synthesized), In (5 N ingot) and Te (5 N chunks). The synthesis of NiTe2 was carried out by heating stoichiometric mixtures of 5 N Ni and Te to 1273 K for 2 hours in evacuated quartz ampoules. This material was powdered and mixed with Cu, In and Te in the ratio corresponding to the stoichiometry Cu1-xNixInTe2 (x = 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 and 0.05). The synthesis of the polycrystalline products were carried out in evacuated (10^-3 Pa), sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, maintained at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 823 K and 70 MPa for 1 h. Compacted disc-shaped samples (diameter 12 mm and thickness ca 2 mm) reached ≥95% of theoretical (X-ray) densities of the prepared compounds. To reach the homogenous distribution of the dopant and to stabilize the physical properties, the hot-pressed samples were annealed in a sealed quartz ampule under an argon atmosphere at T=773 K for 5 days.
Question: Is the experiment solid state synthesis?
Answer: YES 

Document: \"\"\"{document}\"\"\"\nQuestion: Is the experiment solid state synthesis? \nAnswer: '''
# Example prompt 4- 
template_prompt_4=f'''Answer the following question using the document provided in three \".
Answers should be in \"YES\" or \"NO\". If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition Cu1-xNixInTe2 with nominal x values between 0 and 0.05 were prepared utilizing solid-state reactions. Polycrystalline samples were synthesized from a mixture of pure elements and compounds, Cu (4 N shots), NiTe2 (synthesized), In (5 N ingot) and Te (5 N chunks). The synthesis of NiTe2 was carried out by heating stoichiometric mixtures of 5 N Ni and Te to 1273 K for 2 hours in evacuated quartz ampoules. This material was powdered and mixed with Cu, In and Te in the ratio corresponding to the stoichiometry Cu1-xNixInTe2 (x = 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 and 0.05). The synthesis of the polycrystalline products were carried out in evacuated (10^-3 Pa), sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, maintained at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 823 K and 70 MPa for 1 h. Compacted disc-shaped samples (diameter 12 mm and thickness ca 2 mm) reached ≥95% of theoretical (X-ray) densities of the prepared compounds. To reach the homogenous distribution of the dopant and to stabilize the physical properties, the hot-pressed samples were annealed in a sealed quartz ampule under an argon atmosphere at T=773 K for 5 days.
Question: Does the experiment resulted in pure phase formation of crystal? 
Answer: Not Specified 

Document: \"\"\"{document}\"\"\"\nQuestion: Does the experiment resulted in pure phase formation of crystal? \nAnswer: '''
# Example prompt 5- 
template_prompt_5=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition Cu1-xNixInTe2 with nominal x values between 0 and 0.05 were prepared utilizing solid-state reactions. Polycrystalline samples were synthesized from a mixture of pure elements and compounds, Cu (4 N shots), NiTe2 (synthesized), In (5 N ingot) and Te (5 N chunks). The synthesis of NiTe2 was carried out by heating stoichiometric mixtures of 5 N Ni and Te to 1273 K for 2 hours in evacuated quartz ampoules. This material was powdered and mixed with Cu, In and Te in the ratio corresponding to the stoichiometry Cu1-xNixInTe2 (x = 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 and 0.05). The synthesis of the polycrystalline products were carried out in evacuated (10^-3 Pa), sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, maintained at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 823 K and 70 MPa for 1 h. Compacted disc-shaped samples (diameter 12 mm and thickness ca 2 mm) reached ≥95% of theoretical (X-ray) densities of the prepared compounds. To reach the homogenous distribution of the dopant and to stabilize the physical properties, the hot-pressed samples were annealed in a sealed quartz ampule under an argon atmosphere at T=773 K for 5 days.
Question: What is the dopant used in the experiment to dope the base compound?
Answer: Ni

Document: \"\"\"{document}\"\"\"\nQuestion: What is the dopant used in the experiment to dope the base compound? \nAnswer: '''
# Example prompt 5a- 
template_prompt_5a=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition Cu1-xNixInTe2 with nominal x values between 0 and 0.05 were prepared utilizing solid-state reactions. Polycrystalline samples were synthesized from a mixture of pure elements and compounds, Cu (4 N shots), NiTe2 (synthesized), In (5 N ingot) and Te (5 N chunks). The synthesis of NiTe2 was carried out by heating stoichiometric mixtures of 5 N Ni and Te to 1273 K for 2 hours in evacuated quartz ampoules. This material was powdered and mixed with Cu, In and Te in the ratio corresponding to the stoichiometry Cu1-xNixInTe2 (x = 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 and 0.05). The synthesis of the polycrystalline products were carried out in evacuated (10^-3 Pa), sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, maintained at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 823 K and 70 MPa for 1 h. Compacted disc-shaped samples (diameter 12 mm and thickness ca 2 mm) reached ≥95% of theoretical (X-ray) densities of the prepared compounds. To reach the homogenous distribution of the dopant and to stabilize the physical properties, the hot-pressed samples were annealed in a sealed quartz ampule under an argon atmosphere at T=773 K for 5 days.
Question: What is the base compound used in the experiment? Exclude dopant and do not inlude \"x\" when you mention the base compound.
Answer: CuInTe2

Document: \"\"\"{document}\"\"\"\nQuestion: What is the base compound used in the experiment? Exclude dopant and do not inlude \"x\" when you mention the base compound. \nAnswer: '''
# Example prompt 5b- 
template_prompt_5b=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition Cu1-xNixInTe2 with nominal x values between 0 and 0.05 were prepared utilizing solid-state reactions. Polycrystalline samples were synthesized from a mixture of pure elements and compounds, Cu (4 N shots), NiTe2 (synthesized), In (5 N ingot) and Te (5 N chunks). The synthesis of NiTe2 was carried out by heating stoichiometric mixtures of 5 N Ni and Te to 1273 K for 2 hours in evacuated quartz ampoules. This material was powdered and mixed with Cu, In and Te in the ratio corresponding to the stoichiometry Cu1-xNixInTe2 (x = 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04 and 0.05). The synthesis of the polycrystalline products were carried out in evacuated (10^-3 Pa), sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, maintained at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 823 K and 70 MPa for 1 h. Compacted disc-shaped samples (diameter 12 mm and thickness ca 2 mm) reached ≥95% of theoretical (X-ray) densities of the prepared compounds. To reach the homogenous distribution of the dopant and to stabilize the physical properties, the hot-pressed samples were annealed in a sealed quartz ampule under an argon atmosphere at T=773 K for 5 days.\nAnswer to the 2 questions in a single list as ['Answer1','Answer2']
Question 1: What is the base compound used in the experiment? Exclude dopant and do not inlude \"x\" when you mention the base compound.
Question 2: What is the dopant used in the experiment to dope the base compound? Generally, it is written before \"x\". The dopant is not included in base compound. Write chemical symbol (e.g. C for Carbon)
Answer: ['CuInTe2','Ni']

Document: \"\"\"{document}\"\"\"\nAnswer to the 2 questions in a single list as ['Answer1','Answer2']
Question 1: What is the base compound used in the experiment? Exclude dopant and do not inlude \"x\" when you mention the base compound.
Question 2: What is the dopant used in the experiment to dope the base compound? Generally, it is written before \"x\". The dopant is not included in base compound. Write chemical symbol (e.g. C for Carbon)
Answer: '''
# Example prompt 6- 
template_prompt_6=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"NA\". Answer in the following table format.
Temperature | Temperature Unit | Temperature Ramp rate | Temperature Ramp Rate Unit | Duration | Duration Unit
number | unit | number | unit | number | unit

Example 
Document: Polycrystalline samples of the composition CuIn1-xHgxTe2 with nominal x values between 0 and 0.21 have been prepared using a solid-state reaction. Polycrystalline samples were synthesized from mixtures of pure elements, Cu (4N shots), In (5N ingot), Hg (4N), and Te (5N chunks) all Sigma-Aldrich. The synthesis of polycrystalline products was carried out in evacuated sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, kept at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 673 K and 70 MPa for 1 h. Compacted disc shaped samples (diameter 12 mm and thickness ca. 2 mm) reached ≥95% of the theoretical (X-ray) densities of the prepared compounds.
Question: What is the temperature profile of the experiment? Answer in table format.
Answer: 
Temperature | Temperature Unit | Temperature Ramp rate | Temperature Ramp Rate Unit | Duration | Duration Unit
1173 | K | NA | NA | 12 | hour
923 | K | 5 | k/min | 1 | week
673 | K | NA | NA | 1 | hour

Document: \"\"\"{document}\"\"\"\nWhat is the temperature profile of the experiment? Answer in table format. \nAnswer: '''
# Example prompt 6a- 
template_prompt_6a=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition CuIn1-xHgxTe2 with nominal x values between 0 and 0.21 have been prepared using a solid-state reaction. Polycrystalline samples were synthesized from mixtures of pure elements, Cu (4N shots), In (5N ingot), Hg (4N), and Te (5N chunks) all Sigma-Aldrich. The synthesis of polycrystalline products was carried out in evacuated sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, kept at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 673 K and 70 MPa for 1 h. Compacted disc shaped samples (diameter 12 mm and thickness ca. 2 mm) reached ≥95% of the theoretical (X-ray) densities of the prepared compounds.
Question: What is the temperature profile of the experiment? Answer in table format.
Answer: 
Temperature | Temperature Unit | Duration | Duration Unit
1173 | K | 12 | hour
923 | K | 1 | week
673 | K | 1 | hour

Document: \"\"\"{document}\"\"\"\nWhat is the temperature profile of the experiment? Answer in table format. \nAnswer: '''
# Example prompt 6b- 
template_prompt_6b=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition CuIn1-xHgxTe2 with nominal x values between 0 and 0.21 have been prepared using a solid-state reaction. Polycrystalline samples were synthesized from mixtures of pure elements, Cu (4N shots), In (5N ingot), Hg (4N), and Te (5N chunks) all Sigma-Aldrich. The synthesis of polycrystalline products was carried out in evacuated sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, kept at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 673 K and 70 MPa for 1 h. Compacted disc shaped samples (diameter 12 mm and thickness ca. 2 mm) reached ≥95% of the theoretical (X-ray) densities of the prepared compounds.
Question: What is the temperature profile of the experiment? Answer in table format.
Answer: 
Temperature | Temperature Unit | Temperature Ramp rate | Temperature Ramp Rate Unit
1173 | K | NA | NA
923 | K | 5 | k/min
673 | K | NA | NA

Document: \"\"\"{document}\"\"\"\nWhat is the temperature profile of the experiment? Answer in table format. \nAnswer: '''
# Example prompt 6c- 
template_prompt_6c=f'''Check the data in three \".
Does every number and unit has '|' in between? If not incert one.
Temperature | Temperature Unit | Temperature Ramp rate | Temperature Ramp Rate Unit | Duration | Duration Unit
number | unit | number | unit | number | unit

Example 
Data: Temperature | Temperature Unit | Temperature Ramp rate | Temperature Ramp Rate Unit | Duration | Duration Unit
1173 | K | NA | NA | 12 days | 
923 | K | 5 k/min | 1 week | NA |
673 K | NA | NA | 1 | hour
Is the given data following this foramt? If not re-format.
Reformat: Temperature | Temperature Unit | Temperature Ramp rate | Temperature Ramp Rate Unit | Duration | Duration Unit
1173 | K | NA | NA | 12 | day
923 | K | 5 | k/min | 1 | week
673 | K | NA | NA | 1 | hour

Data: \"\"\"{document}\"\"\"\nIs the given data following this foramt? If not re-format. \nReformat: '''
# Example prompt 7- 
template_prompt_7=f'''Answer the following question using the document provided in three \".
If a particular piece of information is not present, output \"NA\".

Example 
Document: Polycrystalline samples of the composition CuIn1-xHgxTe2 with nominal x values between 0 and 0.21 have been prepared using a solid-state reaction. Polycrystalline samples were synthesized from mixtures of pure elements, Cu (4N shots), In (5N ingot), Hg (4N), and Te (5N chunks) all Sigma-Aldrich. The synthesis of polycrystalline products was carried out in evacuated sealed graphitized quartz ampoules. The ampoules were heated to 1173 K over 10 h, kept at this temperature for 12 h, cooled to 923 K at the rate of 5 K/min, annealed at 923 K for one week and finally quenched in air. The products were powdered for 1 min in a vibrating mill under hexane and identified by X-ray diffraction (XRD). The samples for physical measurements were hot-pressed at 673 K and 70 MPa for 1 h. Compacted disc shaped samples (diameter 12 mm and thickness ca. 2 mm) reached ≥95% of the theoretical (X-ray) densities of the prepared compounds.
Question 1: Choose one of the cooling type whether it is left in the room, in water or immerse in something cold, or left in furnace: "Room" or "Quenching" or "Furnace"
Question 2: Choose what is the densification technique used to densify the powder: "Hot Press" or "Sintering" or "NA"
Answer: ['Room', 'Hot Press']

Document: \"\"\"{document}\"\"\"\nWhat is the temperature profile of the experiment? 
Question 1: Choose one of the cooling type whether it is left in the room, in water or immerse in something cold, or left in furnace: "Room" or "Quenching" or "Furnace"
Question 2: Choose what is the densification technique used to densify the powder: "Hot Press" or "Sintering" or "NA"
Answer: '''

#extraction def
def extract_data_v2 (file_name, key):
    
    loader = PyPDFLoader(file_name)
    pages = loader.load_and_split()

    print('Synthesis info:')
    #find chunk with synthesis info
    results = []
    for page in pages:
        results.append(extract_chunk(page.page_content,template_prompt_1, key))
    syn_p = str()
    for ind, i in enumerate(results):
        if 'YES' in i:
            syn_p += str(pages[ind].page_content)
    
    if len(syn_p) == 0:
        return pd.DataFrame()
    else:
        if len(syn_p) > 10000:
            syn_p = syn_p[:10000]
        print('\nPhase info:')
        #check if it is pure phase
        results = []
        for page in pages:
            results.append(extract_chunk(page.page_content,template_prompt_4, key))
        pure_phase_formed = 0
        for ind, i in enumerate(results):
            if 'YES' in i:
                pure_phase_formed = 1

        print('\nBase & Dopant info:')
        #find dopant
        results = []
        for page in pages:
            results.append(extract_chunk(page.page_content,template_prompt_5b, key))

        base_compound = []
        dopant = []

        for i in results:
            res = (i.strip('[]').replace("'","").replace(" ","").split(','))
            base_compound+=[res[0]]
            dopant+=[res[1]]

        counter = collections.Counter(base_compound)
        del counter['NA']
        base_compound = 'NA'
        if len(counter.most_common()) > 0:
            base_compound = counter.most_common()[0][0]
        counter = collections.Counter(dopant)
        del counter['NA']
        dopant = 'NA'
        if len(counter.most_common()) > 0:
            dopant = counter.most_common()[0][0]     

        print('\nTemp Profile:')
        #find temperature profile
        result = extract_chunk(syn_p,template_prompt_6, key)

        print('\nCheck Profile format:')
        result = extract_chunk(result,template_prompt_6c, key)
        temperatures = []
        times = []
        rates = []
        for tp in result.split('\n')[1:]:
            tp = tp.replace(" ", "")
            temp = tp.split('|')
            if not ('NA' in str(temp[0])):
                if 'C' in temp[1]:
                    temperatures += [int(temp[0])+273]
                else:
                    temperatures += [int(temp[0])]

                if not ('NA' in str(temp[4])):
                    if 'week' in temp[5]:
                        times += [float(temp[4])*24*7]
                    elif 'day' in temp[5]:
                        times += [float(temp[4])*24]
                    elif ('hour' in temp[5]) or ('hr' in temp[5]) :
                        times += [float(temp[4])]
                    elif ('min' in temp[5]):
                        times += [float(temp[4])/60]
                    else:
                        raise ValueError('Value: '+ str(temp[4]) + '! Check Time Unit Conditions.')
                else:
                    times += ['NA']

                if not ('NA' in str(temp[2])):
                    if 'min' in temp[3]:
                        rates += [float(temp[2])*60]
                    elif 'd' in temp[3]:
                        rates += [float(temp[2])/24]
                    elif 'h' in temp[3]:
                        rates += [float(temp[2])]
                    else:
                        raise ValueError('Value: '+ str(temp[2]) + '! Check Rate Unit Conditions.')
                else:
                    rates += ['NA']  
        try:
            max_ind = temperatures.index(max(temperatures))
        except:
            return pd.DataFrame()
        len_temp = len(temperatures)

        prim_temp = 298
        prim_time = 'NA'
        prim_rate = 'NA'
        sec_temp = max(temperatures)
        sec_time = times[max_ind]
        sec_rate = rates[max_ind]
        anne_temp = 298
        anne_time = 'NA'
        anne_rate = 'NA'

        if not (max_ind == 0):
            prim_temp = temperatures[max_ind-1]
            prim_time = times[max_ind-1]
            prim_rate = rates[max_ind-1]
        if not (max_ind == (len_temp-1)):
            anne_temp = temperatures[max_ind+1]
            anne_time = times[max_ind+1]
            anne_rate = rates[max_ind+1]


        print('\nCooling & Densification:')
        #find cooling type and densification method
        result = extract_chunk(syn_p,template_prompt_7, key)
        result = result.strip("[]").replace("'","").replace(" ", "").split(',')

        cooling_method = result[0]
        densifi_method = result[1]

        data = pd.DataFrame([{'file_name': file_name,
                  'base_compound': base_compound,
                  'dopant': dopant,
                  'primiary_temperature': prim_temp,
                  'primiary_duration': prim_time,
                  'primiary_ramp_rate': prim_rate,
                  'secondary_temperature': sec_temp,
                  'secondary_duration': sec_time,
                  'secondary_ramp_rate': sec_rate,
                  'annealing_temperature': anne_temp,
                  'annealing_duration': anne_time,
                  'annealing_ramp_rate': anne_rate,
                  'cooling_method': cooling_method,
                  'densification': densifi_method,
                  'pure_phase': pure_phase_formed
                 }])

        return data