const state = {
  auth: { username: "student", password: "student123" },
  profile: null,
  sessionId: null,
  sending: false,
  modelName: "mock"
};

const els = {
  serviceState: document.querySelector("#serviceState"),
  modelState: document.querySelector("#modelState"),
  loginForm: document.querySelector("#loginForm"),
  username: document.querySelector("#username"),
  password: document.querySelector("#password"),
  loginState: document.querySelector("#loginState"),
  accountBadge: document.querySelector("#accountBadge"),
  activeAccount: document.querySelector("#activeAccount"),
  activeRole: document.querySelector("#activeRole"),
  switchAccount: document.querySelector("#switchAccount"),
  studentView: document.querySelector("#studentView"),
  adminView: document.querySelector("#adminView"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  newSession: document.querySelector("#newSession"),
  sessionBadge: document.querySelector("#sessionBadge"),
  reports: document.querySelector("#reports"),
  refreshAdmin: document.querySelector("#refreshAdmin"),
  metricReports: document.querySelector("#metricReports"),
  metricHigh: document.querySelector("#metricHigh"),
  metricExcel: document.querySelector("#metricExcel"),
  metricAlerts: document.querySelector("#metricAlerts"),
  knowledgeUploadForm: document.querySelector("#knowledgeUploadForm"),
  knowledgeFile: document.querySelector("#knowledgeFile"),
  knowledgeUploadState: document.querySelector("#knowledgeUploadState")
};

function authHeader() {
  return `Basic ${btoa(`${state.auth.username}:${state.auth.password}`)}`;
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}), Authorization: authHeader() };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response;
}

function setPill(el, text, tone = "ok") {
  el.textContent = text;
  el.className = `pill ${tone}`;
}

function isAdmin(profile) {
  return profile.roles?.some((role) => role.authority === "ROLE_ADMIN");
}

async function checkHealth() {
  try {
    const response = await fetch("/actuator/health");
    const body = await response.json();
    setPill(els.serviceState, body.status === "UP" ? "服务正常" : `服务 ${body.status}`, body.status === "UP" ? "ok" : "danger");
  } catch {
    setPill(els.serviceState, "服务 DOWN", "danger");
  }
}

async function loadAgentStatus() {
  const response = await api("/api/agent/status");
  const status = await response.json();
  state.modelName = status.model || "mock";
  if (status.realModelEnabled) {
    setPill(els.modelState, `${status.provider} / ${displayModel(state.modelName)}`, "ok");
  } else {
    setPill(els.modelState, "mock 演示", "warn");
  }
}

function displayModel(model) {
  return model || "未配置模型";
}

async function login(event) {
  event?.preventDefault();
  state.auth.username = els.username.value.trim();
  state.auth.password = els.password.value;
  try {
    const response = await api("/api/profile");
    state.profile = await response.json();
    await loadAgentStatus();
    showApp();
    els.loginState.textContent = "登录成功";
    if (isAdmin(state.profile)) {
      await loadAdminDashboard();
    }
  } catch (error) {
    els.loginState.textContent = `登录失败：${error.message}`;
  }
}

function showApp() {
  const admin = isAdmin(state.profile);
  els.accountBadge.hidden = false;
  els.switchAccount.hidden = false;
  els.activeAccount.textContent = state.profile.displayName || state.profile.username;
  els.activeRole.textContent = admin ? "管理员账号" : "学生账号";
  els.loginForm.hidden = true;
  els.studentView.hidden = admin;
  els.adminView.hidden = !admin;
}

function logout() {
  state.profile = null;
  state.sessionId = null;
  els.accountBadge.hidden = true;
  els.switchAccount.hidden = true;
  els.loginForm.hidden = false;
  els.studentView.hidden = false;
  els.adminView.hidden = true;
  setPill(els.sessionBadge, "READY");
}

function clearWelcome() {
  const empty = els.messages.querySelector(".empty");
  if (empty) empty.remove();
}

function addMessage(role, content) {
  clearWelcome();
  const row = document.createElement("article");
  row.className = `message ${role}`;
  row.innerHTML = `
    <div class="message-role">${role === "user" ? "我" : "MindBridge"}</div>
    <div class="bubble"></div>
  `;
  row.querySelector(".bubble").textContent = content;
  els.messages.append(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return row.querySelector(".bubble");
}

function parseSse(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const rest = parts.pop();
  for (const part of parts) {
    const dataLine = part.split("\n").find((line) => line.startsWith("data: "));
    if (!dataLine) continue;
    onEvent(JSON.parse(dataLine.slice(6)));
  }
  return rest;
}

async function sendMessage(event) {
  event.preventDefault();
  if (state.sending || isAdmin(state.profile)) return;
  const message = els.messageInput.value.trim();
  if (!message) return;
  state.sending = true;
  els.sendButton.disabled = true;
  setPill(els.sessionBadge, "THINKING", "warn");
  els.messageInput.value = "";
  addMessage("user", message);
  const assistant = addMessage("assistant", "");
  let raw = "";

  try {
    const response = await api("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId: state.sessionId, message })
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let streamFailed = false;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSse(buffer, (eventData) => {
        if (eventData.type === "meta") state.sessionId = eventData.sessionId;
        if (eventData.type === "token") {
          raw += eventData.content || "";
          assistant.textContent = raw;
          els.messages.scrollTop = els.messages.scrollHeight;
        }
        if (eventData.type === "error") {
          streamFailed = true;
          if (!raw) assistant.textContent = eventData.message || "MCP 工具调用失败";
          setPill(els.sessionBadge, "ERROR", "danger");
        }
      });
    }
    if (!streamFailed) setPill(els.sessionBadge, "DONE", "ok");
  } catch (error) {
    assistant.textContent = `发送失败：${error.message}`;
    setPill(els.sessionBadge, "ERROR", "danger");
  } finally {
    state.sending = false;
    els.sendButton.disabled = false;
  }
}

async function loadAdminDashboard() {
  const [reportsRes, excelRes, alertsRes] = await Promise.all([
    api("/api/admin/reports"),
    api("/api/admin/excel-records"),
    api("/api/admin/alerts")
  ]);
  const reports = await reportsRes.json();
  const excel = await excelRes.json();
  const alerts = await alertsRes.json();
  els.metricReports.textContent = reports.length;
  els.metricHigh.textContent = reports.filter((item) => item.riskLevel === "HIGH").length;
  els.metricExcel.textContent = excel.length;
  els.metricAlerts.textContent = alerts.length;
  renderReports(reports);
}

function renderReports(reports) {
  els.reports.innerHTML = "";
  if (!reports.length) {
    els.reports.innerHTML = `<div class="empty small"><strong>暂无报告</strong><p>学生咨询或风险场景会在这里沉淀记录。</p></div>`;
    return;
  }
  for (const item of reports) {
    const card = document.createElement("article");
    card.className = `report risk-${item.riskLevel.toLowerCase()}`;
    card.innerHTML = `
      <div>
        <strong>${item.displayName} · ${item.riskLevel}</strong>
        <span>${new Date(item.createdAt).toLocaleString()}</span>
      </div>
      <p>${item.summary}</p>
      <small>${item.content}</small>
    `;
    els.reports.append(card);
  }
}

async function uploadKnowledgeFile(event) {
  event.preventDefault();
  const file = els.knowledgeFile.files?.[0];
  if (!file) {
    els.knowledgeUploadState.textContent = "请先选择文件";
    return;
  }
  const data = new FormData();
  data.append("file", file);
  els.knowledgeUploadState.textContent = "正在切分入库...";
  try {
    const response = await api("/api/admin/knowledge/file", { method: "POST", body: data });
    const result = await response.json();
    els.knowledgeUploadState.textContent = `${result.source} 已入库 ${result.chunks} 个片段`;
    els.knowledgeFile.value = "";
  } catch (error) {
    els.knowledgeUploadState.textContent = `上传失败：${error.message}`;
  }
}

document.querySelectorAll("[data-quick]").forEach((button) => {
  button.addEventListener("click", () => {
    els.messageInput.value = button.dataset.quick;
    els.messageInput.focus();
  });
});
els.loginForm.addEventListener("submit", login);
els.chatForm.addEventListener("submit", sendMessage);
els.newSession.addEventListener("click", () => {
  state.sessionId = null;
  els.messages.innerHTML = `<div class="empty"><strong>新会话已开始</strong><p>你可以继续输入新的问题。</p></div>`;
  setPill(els.sessionBadge, "READY");
});
els.switchAccount.addEventListener("click", logout);
els.refreshAdmin?.addEventListener("click", loadAdminDashboard);
els.knowledgeUploadForm?.addEventListener("submit", uploadKnowledgeFile);

checkHealth();
login();
