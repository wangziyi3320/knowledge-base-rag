const documentList = document.querySelector('#documentList')
const fileInput = document.querySelector('#fileInput')
const uploadStatus = document.querySelector('#uploadStatus')
const queryForm = document.querySelector('#queryForm')
const queryStatus = document.querySelector('#queryStatus')
const answer = document.querySelector('#answer')
const citations = document.querySelector('#citations')

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
  queryStatus.textContent = `回答模式：${data.generation_mode}`
  answer.textContent = data.answer
  answer.classList.remove('hidden')
  citations.innerHTML = data.citations.map((item) => `
    <section class="citation">
      <h3>[${item.rank}] ${escapeHtml(item.document_name)} · 相关度 ${item.score}</h3>
      <p>${escapeHtml(item.content)}</p>
    </section>`).join('')
})

loadDocuments().catch(() => { documentList.innerHTML = '<p class="status">文档列表加载失败。</p>' })

