from ollama import ProcessResponse, chat, ps, pull

# Ensure at least one model is loaded
response = pull('deepseek-r1', stream=True)
progress_states = set()
for progress in response:
  if progress.get('status') in progress_states:
    continue
  progress_states.add(progress.get('status'))
  print(progress.get('status'))

print('\n')

print('Waiting for model to load... \n')
chat(model='deepseek-r1', messages=[{'role': 'user', 'content': 'what is capital of india?'}])


response: ProcessResponse = ps()
for model in response.models:
  print('Model: ', model.model)
  print('  Digest: ', model.digest)
  print('  Expires at: ', model.expires_at)
  print('  Size: ', model.size)
  print('  Size vram: ', model.size_vram)
  print('  Details: ', model.details)
  print('\n')
