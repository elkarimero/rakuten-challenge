from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "Translate the following text into french: I’ve never seen a platform as easy to use"}
  ]
)

print(completion.choices[0].message);
