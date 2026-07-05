(() => {
  "use strict";

  const form = document.getElementById("composer");
  const input = document.getElementById("message-input");
  const messages = document.getElementById("messages");
  const typing = document.getElementById("typing");

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Very small markdown-ish renderer: paragraphs, **bold**, `code`.
  // The assistant's answers are plain text/light markdown, not full
  // HTML, so this is deliberately minimal rather than pulling in a
  // markdown dependency for a single chat view.
  function renderContent(text) {
    const escaped = escapeHtml(text);
    const withInline = escaped
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
    return withInline
      .split(/\n{2,}/)
      .map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`)
      .join("");
  }

  function addBubble(text, role) {
    const bubble = document.createElement("div");
    bubble.className = `bubble bubble-${role}`;
    if (role !== "error") {
      const glare = document.createElement("div");
      glare.className = "bubble-glare";
      bubble.appendChild(glare);
    }
    const content = document.createElement("div");
    content.innerHTML = renderContent(text);
    bubble.appendChild(content);
    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }

  function setBusy(busy) {
    input.disabled = busy;
    form.querySelector("button").disabled = busy;
    typing.classList.toggle("hidden", !busy);
    if (busy) {
      messages.scrollTop = messages.scrollHeight;
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    addBubble(message, "user");
    input.value = "";
    setBusy(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      if (!response.ok) {
        addBubble(data.error || "Something went wrong.", "error");
      } else {
        addBubble(data.reply, "assistant");
      }
    } catch (err) {
      addBubble("Could not reach the assistant server.", "error");
    } finally {
      setBusy(false);
      input.focus();
    }
  });

  input.focus();

  // ---------------------------------------------------------------
  // Settings panel
  // ---------------------------------------------------------------

  const settingsButton = document.getElementById("settings-button");
  const settingsOverlay = document.getElementById("settings-overlay");
  const settingsClose = document.getElementById("settings-close");
  const settingsForm = document.getElementById("settings-form");
  const settingsError = document.getElementById("settings-error");

  function humanizeFieldName(name) {
    return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function inputTypeFor(kind) {
    return kind === "str" ? "text" : "number";
  }

  async function openSettings() {
    settingsError.classList.add("hidden");
    settingsForm.innerHTML = "<p>Loading…</p>";
    settingsOverlay.classList.remove("hidden");

    try {
      const response = await fetch("/api/config");
      const data = await response.json();
      renderSettingsForm(data);
    } catch (err) {
      settingsForm.innerHTML = "";
      settingsError.textContent = "Could not load settings.";
      settingsError.classList.remove("hidden");
    }
  }

  function renderSettingsForm(data) {
    settingsForm.innerHTML = "";
    const overridden = new Set(data.overridden || []);

    for (const [name, kind] of Object.entries(data.fields)) {
      const field = document.createElement("div");
      field.className = "settings-field";

      const label = document.createElement("label");
      label.setAttribute("for", `setting-${name}`);
      label.textContent = humanizeFieldName(name);
      if (overridden.has(name)) {
        const dot = document.createElement("span");
        dot.className = "override-dot";
        dot.title = "Overridden in config.json";
        label.appendChild(dot);
      }

      const input = document.createElement("input");
      input.type = inputTypeFor(kind);
      if (kind === "float") input.step = "any";
      input.id = `setting-${name}`;
      input.name = name;
      input.value = data.values[name];

      field.appendChild(label);
      field.appendChild(input);
      settingsForm.appendChild(field);
    }
  }

  function closeSettings() {
    settingsOverlay.classList.add("hidden");
  }

  settingsButton.addEventListener("click", openSettings);
  settingsClose.addEventListener("click", closeSettings);
  settingsOverlay.addEventListener("click", (event) => {
    if (event.target === settingsOverlay) closeSettings();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !settingsOverlay.classList.contains("hidden")) {
      closeSettings();
    }
  });

  settingsForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    settingsError.classList.add("hidden");

    const values = {};
    for (const el of settingsForm.querySelectorAll("input")) {
      values[el.name] = el.value;
    }

    try {
      const response = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      const data = await response.json();
      if (!response.ok) {
        const details = data.details
          ? Object.entries(data.details).map(([k, v]) => `${k}: ${v}`).join(", ")
          : data.error;
        settingsError.textContent = details || "Could not save settings.";
        settingsError.classList.remove("hidden");
        return;
      }
      closeSettings();
    } catch (err) {
      settingsError.textContent = "Could not reach the assistant server.";
      settingsError.classList.remove("hidden");
    }
  });
})();
