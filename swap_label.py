import csv
import re

def swap_answer(answer):

    if "nothing has changed" in answer.lower():
        return answer


    finding_pattern = re.compile(r"the main image (has (an )?additional|is missing the) findings? of (.*?) than the reference image", re.IGNORECASE)


    new_additional_diseases = []
    new_missing_diseases = []
    swapped_severity_sentences = []
    other_sentences = []

  
    sentences = [s.strip() for s in answer.split('.') if s.strip()]

    for sentence in sentences:
        
        match_finding = finding_pattern.search(sentence)
        if match_finding:
            finding_type = match_finding.group(1) 
            disease_string = match_finding.group(3).strip() 
            diseases = [d.strip() for d in re.split(r',| and ', disease_string) if d.strip()]

            if 'additional' in finding_type.lower():
              
                new_missing_diseases.extend(diseases)
            else:  
               
                new_additional_diseases.extend(diseases)
            continue

      
        if " has changed from " in sentence.lower():
      
            parts = re.split(r' from ', sentence, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                prefix = parts[0] + " from "
               
                severities = [s.strip() for s in re.split(r' to ', parts[1], flags=re.IGNORECASE)]
                
                severities.reverse()
              
                reversed_severities_str = " to ".join(severities)
                swapped_severity_sentences.append(prefix + reversed_severities_str)
                continue
        
      
        other_sentences.append(sentence)

  
    final_parts = other_sentences + swapped_severity_sentences
    
    if new_additional_diseases:
        if len(new_additional_diseases) == 1:
            diseases_str = new_additional_diseases[0]
        else:
     
            diseases_str = f"{', '.join(new_additional_diseases[:-1])}, and {new_additional_diseases[-1]}"
        
 
        if len(new_additional_diseases) == 1:
            final_parts.append(f"the main image has an additional finding of {diseases_str} than the reference image")
        else: 
            final_parts.append(f"the main image has additional findings of {diseases_str} than the reference image")

    if new_missing_diseases:
        if len(new_missing_diseases) == 1:
            diseases_str = new_missing_diseases[0]
        else:
           
            diseases_str = f"{', '.join(new_missing_diseases[:-1])}, and {new_missing_diseases[-1]}"

       
        if len(new_missing_diseases) == 1:
            final_parts.append(f"the main image is missing the finding of {diseases_str} than the reference image")
        else: 
            final_parts.append(f"the main image is missing the findings of {diseases_str} than the reference image")

    if not final_parts:
        return "nothing has changed."
        
    return ". ".join(final_parts) + "."

def process_csv(input_filename, output_filename):
    """
    CSV 파일을 읽어 이미지 순서와 답변을 수정한 후 새 CSV 파일에 저장합니다.
    """
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
      
            try:
                header = next(reader)
                writer.writerow(header)
            except StopIteration:
                print(f"오류: 입력 파일 '{input_filename}'이 비어있습니다.")
                return

       
            for row in reader:
                if len(row) < 3:
                    continue 
                
             
                images = row[0].split(',')
                if len(images) == 2:
                    swapped_images = f"{images[1].strip()},{images[0].strip()}"
                else:
                    swapped_images = row[0]

          
                question = row[1]
                
            
                original_answer = row[2]
                swapped_ans = swap_answer(original_answer)
                
    
                writer.writerow([swapped_images, question, swapped_ans])
        
        print(f"'{input_filename}' 파일 처리가 완료되었습니다. 결과는 '{output_filename}'에 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_filename}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {e}")


if __name__ == '__main__':

    input_csv_file = 'formatted_for_train.csv'
   
    output_csv_file = 'swapped_train.csv'

    
    try:
        with open(input_csv_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"'{input_csv_file}' 파일을 찾을 수 없어 예제 파일을 생성합니다.")
        sample_data = """images,question,answer
"p18/p18550032/s51679419/98893daa-f4632ce8-7ed2a02b-414f6f9d-8ba34213.jpg,p18/p18550032/s55937951/4de21995-9472a06-acdc3b23-b4465995-acf4f42b.jpg",what has changed compared to the reference image?,the level of pleural effusion has changed from small to mild to moderate. the main image has an additional finding of cardiomegaly than the reference image. the main image is missing the finding of vascular congestion than the reference image.
"p18/p18431965/s52871289/700a33da-9a946ed2-f4647307-38d67f64-bd8618a8.jpg,p18/p18431965/s53481230/70d7c13a-042c002b-e7ca3d36-4683aaa0-03f4e20f.jpg",what has changed compared to the reference image?,"the main image has additional findings of pleural effusion, edema, and cardiomegaly than the reference image. the main image is missing the finding of pneumonia than the reference image. "
"p18/p18879976/s54522940/2034e94d-d028fc46-dc29d156-b6f061bc-edc32f8f.jpg,p18/p18879976/s58724539/03ba926d-fe132aa7-8a8e3a6b-423018b4-c2d2660.jpg",what has changed compared to the reference image?,nothing has changed.
"img_A.jpg,img_B.jpg",what has changed?,"the level of pleural effusion has changed from small to small moderate . the level of cardiomegaly has changed from moderate to severe to moderate . the level of edema has changed from moderate to moderately severe ."
"p15/p15475850/s53471838/93bac912-3355546d-e03d762e-be1513e2-225f036c.jpg,p15/p15475850/s54266505/1df37674-7c74402d-2896d113-2ee5ae9b-46ba2050.jpg",what has changed compared to the reference image?,"the main image has additional findings of vascular congestion, and atelectasis than the reference image. the main image is missing the findings of pleural effusion, and consolidation than the reference image. "
"""
        with open(input_csv_file, 'w', encoding='utf-8', newline='') as f:
            f.write(sample_data)


    process_csv(input_csv_file, output_csv_file)

