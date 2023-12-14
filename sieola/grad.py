import gradio as gr
import pandas as pd 
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

df = pd.read_excel(r'C:\Users\Ameer\pandas\maxbyte\interns\sieola\production_report.xlsx',sheet_name='Sheet2',engine = 'openpyxl')

llm = OpenAI(api_token="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")

sdf = SmartDataframe(df, config={"llm": llm})



# # def greet(name):
# #     return "Hello " + name + "!"

# # demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# # demo.launch()   



def chat(question):
    return sdf.chat(question)

demo = gr.Interface(fn=chat, inputs="text", outputs="text")
demo.launch(share = True)   



