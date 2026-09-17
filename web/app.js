const form = document.querySelector('#composer');
const input = document.querySelector('#message');
const chat = document.querySelector('#chat');
const welcome = document.querySelector('#welcome');
const statusText = document.querySelector('#status-text');
const modelMeta = document.querySelector('#model-meta');
let conversationId = null;

async function refreshHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    const providerName = data.provider === 'ollama' ? 'Ollama' : (data.local_model_loaded ? 'Local scratch model' : 'Offline');
    const modelLabel = data.model ? `Model: ${data.model}` : 'Model: Astra';
    if (modelMeta) {
      modelMeta.textContent = `${modelLabel} · ${providerName}`;
    }
    if (statusText) {
      statusText.textContent = data.ollama_available ? 'Ollama online' : 'Local fallback active';
    }
    const dot = document.querySelector('.status-dot');
    if (dot) dot.style.background = data.ollama_available ? '#3eaa69' : '#d0952d';
  } catch (error) {
    if (statusText) statusText.textContent = 'Status unavailable';
  }
}

refreshHealth();

function addMessage(role, text) {
  const row = document.createElement('div');
  row.className = `message ${role}`;
  row.innerHTML = `<div class="avatar">${role === 'user' ? 'Y' : 'A'}</div><div class="message-content"></div>`;
  row.querySelector('.message-content').textContent = text;
  chat.appendChild(row);
  row.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return row;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  welcome?.remove();
  input.value = '';
  addMessage('user', message);
  const pending = addMessage('assistant', 'Thinking...');
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message, conversation_id: conversationId, temperature: 0.35}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Request failed');
    conversationId = data.conversation_id;
    pending.querySelector('.message-content').textContent = data.answer;
    await refreshHealth();
  } catch (error) {
    pending.querySelector('.message-content').textContent = `AI request failed: ${error.message}`;
  }
});

document.querySelector('#new-chat').addEventListener('click', () => { conversationId = null; chat.innerHTML = ''; chat.appendChild(welcome); });
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 120)}px`; });
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});