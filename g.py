import google.generativeai as genai

genai.configure(api_key="AIzaSyADG5FWkseqATlmFvTRN2b7A7-EEbinLpA")

model = genai.GenerativeModel("models/gemini-1.5-pro")
response = model.generate_content("Explain this telecom issue: network packet loss in VoIP calls.")
print(response.text)
