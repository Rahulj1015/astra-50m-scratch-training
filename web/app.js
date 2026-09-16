const form = document.querySelector('#composer');
const input = document.querySelector('#message');
const chat = document.querySelector('#chat');
const welcome = document.querySelector('#welcome');
let conversationId = null;

fetch('/api/health')
  .then((response) => response.json())
  .then((data) => {
    const meta = document.querySelector('#model-meta');
    if (meta && data.parameters) {
      meta.textContent = `Model: Astra · ${(data.parameters / 1000000).toFixed(1)}M params`;
    }
  })
  .catch(() => {});

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
  const pending = addMessage('assistant', 'Thinking locally...');
  try {
    const response = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message, conversation_id: conversationId, temperature: 0.35}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Request failed');
    conversationId = data.conversation_id;
    pending.querySelector('.message-content').textContent = data.answer;
  } catch (error) {
    pending.querySelector('.message-content').textContent = `Local model error: ${error.message}`;
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