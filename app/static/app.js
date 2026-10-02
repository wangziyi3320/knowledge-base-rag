const documentList = document.querySelector('#documentList')
const fileInput = document.querySelector('#fileInput')
const uploadStatus = document.querySelector('#uploadStatus')
const queryForm = document.querySelector('#queryForm')
const queryStatus = document.querySelector('#queryStatus')
const answer = document.querySelector('#answer')
const citations = document.querySelector('#citations')
const citationHeading = document.querySelector('#citationHeading')
const draftTicket = document.querySelector('#draftTicket')
const ticketForm = document.querySelector('#ticketForm')
const ticketStatus = document.querySelector('#ticketStatus')
const ticketList = document.querySelector('#ticketList')

const escapeHtml = (value) => value.replace(/[&<>'"]/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
})[character])

async function loadDocuments() {
  const response = await fetch('/api/documents')
  const documents = await response.json()
  documentList.innerHTML = documents.length
    ? documents.map((item) => `
      <div class="document">
        <div><strong>${escapeHtml(item.name)}</strong><small>${item.chunk_count} 个片段 · ${item.created_at}</small></div>
        <button data-delete="${item.id}">删除</button>
      </div>`).join('')
    : '<p class="status">尚未上传文档。</p>'
}

fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0]
  if (!file) return
  uploadStatus.textContent = '正在解析和建立索引…'
  const body = new FormData()
  body.append('file', file)
  const response = await fetch('/api/documents', { method: 'POST', body })
  const data = await response.json()
  uploadStatus.textContent = response.ok
    ? (data.duplicate ? '该文档已存在，未重复建立索引。' : `已生成 ${data.chunk_count} 个知识片段。`)
    : (data.detail || '上传失败。')
  fileInput.value = ''
  await loadDocuments()
})

documentList.addEventListener('click', async (event) => {
  const id = event.target.dataset.delete
  if (!id || !window.confirm('确定删除该文档及其索引吗？')) return
  await fetch(`/api/documents/${id}`, { method: 'DELETE' })
  await loadDocuments()
})

queryForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  queryStatus.textContent = '正在检索知识库…'
  answer.classList.add('hidden')
  citationHeading.classList.add('hidden')
  citations.innerHTML = ''
  const response = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: document.querySelector('#question').value,
      top_k: Number(document.querySelector('#topK').value),
    }),
  })
  const data = await response.json()
  if (!response.ok) {
    queryStatus.textContent = data.detail || '查询失败。'
    return
  }
  const modeNames = {
    llm: '模型生成',
    'retrieval-only': '仅展示检索片段',
    'insufficient-evidence': '证据不足',
    'unverified-generation': '模型引用未通过检查',
  }
  queryStatus.textContent = `${modeNames[data.generation_mode] || data.generation_mode} · ${data.evidence_reason}`
  answer.textContent = data.answer
  answer.classList.remove('hidden')
  citationHeading.textContent = data.answerable ? '检索候选片段（请核对引用）' : '检索候选（尚不能作为回答依据）'
  citationHeading.classList.toggle('hidden', !data.citations.length)
  citations.innerHTML = data.citations.map((item) => `
    <section class="citation">
      <h3>[${item.rank}] ${escapeHtml(item.document_name)} · 相对排序分 ${item.score}</h3>
      <p>${escapeHtml(item.content)}</p>
    </section>`).join('')
})

loadDocuments().catch(() => { documentList.innerHTML = '<p class="status">文档列表加载失败。</p>' })

async function loadTickets() {
  const response = await fetch('/api/tickets')
  const tickets = await response.json()
  ticketList.innerHTML = tickets.length
    ? tickets.map((item) => `
      <div class="document">
        <div><strong>#${item.id} ${escapeHtml(item.title)}</strong>
          <small>${escapeHtml(item.category)} · ${escapeHtml(item.status)} · ${escapeHtml(item.created_at)}</small></div>
      </div>`).join('')
    : '<p class="status">暂无已确认的工单。</p>'
}

draftTicket.addEventListener('click', async () => {
  const question = document.querySelector('#question').value.trim()
  if (question.length < 2) {
    ticketStatus.textContent = '请先在知识问答中填写问题。'
    return
  }
  const response = await fetch('/api/tickets/draft', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!response.ok) {
    ticketStatus.textContent = '草稿生成失败。'
    return
  }
  const draft = await response.json()
  document.querySelector('#ticketTitle').value = draft.title
  document.querySelector('#ticketDescription').value = draft.description
  document.querySelector('#ticketCategory').value = draft.category
  document.querySelector('#ticketConfirm').checked = false
  ticketForm.classList.remove('hidden')
  ticketStatus.textContent = '草稿尚未保存，请核对内容。'
})

ticketForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  const response = await fetch('/api/tickets', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title: document.querySelector('#ticketTitle').value,
      description: document.querySelector('#ticketDescription').value,
      category: document.querySelector('#ticketCategory').value,
      confirmed: document.querySelector('#ticketConfirm').checked,
    }),
  })
  if (!response.ok) {
    ticketStatus.textContent = '保存失败，请检查标题、描述和确认状态。'
    return
  }
  ticketForm.classList.add('hidden')
  ticketStatus.textContent = '已确认并保存工单。'
  await loadTickets()
})

loadTickets().catch(() => { ticketList.innerHTML = '<p class="status">工单列表加载失败。</p>' })
