import google.generativeai as genai
genai.configure(api_key="AIzaSyD1x6DsTlcnCd_B3H5DFpC3f6B0o2-mqvQ")
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Hello Gemini!")
print(response.text)
