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

  // Small line-oriented markdown-ish renderer: headers, bullet/numbered
  // lists, paragraphs, **bold**, `code`. The assistant's answers are
  // plain text/light markdown, not full HTML, so this is deliberately
  // minimal rather than pulling in a markdown dependency for a single
  // chat view -- but it needs to at least handle the block-level
  // markdown the model actually produces (headers, lists), or those
  // render as literal "##"/"-" characters instead of formatted text.
  function renderInline(escapedLine) {
    return escapedLine
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");
  }

  function renderContent(text) {
    const lines = escapeHtml(text).split("\n");
    const htmlParts = [];
    let paragraphLines = [];
    let listItems = [];
    let listTag = null; // "ul" | "ol"

    const flushParagraph = () => {
      if (paragraphLines.length) {
        htmlParts.push(`<p>${paragraphLines.join("<br>")}</p>`);
        paragraphLines = [];
      }
    };
    const flushList = () => {
      if (listItems.length) {
        htmlParts.push(`<${listTag}>${listItems.map((item) => `<li>${item}</li>`).join("")}</${listTag}>`);
        listItems = [];
        listTag = null;
      }
    };

    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        flushParagraph();
        flushList();
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        flushParagraph();
        flushList();
        const level = Math.min(heading[1].length + 2, 6); // keep subordinate to the bubble's own heading
        htmlParts.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        continue;
      }

      const bullet = line.match(/^[-*]\s+(.*)$/);
      if (bullet) {
        flushParagraph();
        if (listTag && listTag !== "ul") flushList();
        listTag = "ul";
        listItems.push(renderInline(bullet[1]));
        continue;
      }

      const numbered = line.match(/^\d+\.\s+(.*)$/);
      if (numbered) {
        flushParagraph();
        if (listTag && listTag !== "ol") flushList();
        listTag = "ol";
        listItems.push(renderInline(numbered[1]));
        continue;
      }

      flushList();
      paragraphLines.push(renderInline(line));
    }
    flushParagraph();
    flushList();
    return htmlParts.join("");
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

  // Streams /api/chat/stream's Server-Sent Events: a plain-text
  // "live" bubble grows token-by-token while the model is generating
  // (same idea as the CLI's rich.Live preview), tool calls show as a
  // small status line instead of raw JSON, and once the turn produces
  // a final answer the live bubble is swapped for a properly
  // markdown-rendered one via addBubble().
  async function sendStreaming(message) {
    let liveBubble = null;
    let liveText = "";

    const ensureLiveBubble = () => {
      if (!liveBubble) {
        liveBubble = document.createElement("div");
        liveBubble.className = "bubble bubble-assistant bubble-live";
        const glare = document.createElement("div");
        glare.className = "bubble-glare";
        liveBubble.appendChild(glare);
        const content = document.createElement("div");
        content.className = "live-content";
        liveBubble.appendChild(content);
        messages.appendChild(liveBubble);
        messages.scrollTop = messages.scrollHeight;
      }
      return liveBubble.querySelector(".live-content");
    };

    const addStatusLine = (text) => {
      const line = document.createElement("div");
      line.className = "status-line";
      line.textContent = text;
      messages.appendChild(line);
      messages.scrollTop = messages.scrollHeight;
    };

    const cleanupLive = () => {
      if (liveBubble) {
        liveBubble.remove();
        liveBubble = null;
        liveText = "";
      }
    };

    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok || !response.body) {
      let errorText = "Something went wrong.";
      try {
        const data = await response.json();
        errorText = data.error || errorText;
      } catch (_err) {
        /* response wasn't JSON -- keep the generic message */
      }
      addBubble(errorText, "error");
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        if (!frame.startsWith("data: ")) continue;

        let payload;
        try {
          payload = JSON.parse(frame.slice(6));
        } catch (_err) {
          continue;
        }

        if (payload.type === "token") {
          liveText += payload.text;
          ensureLiveBubble().textContent = liveText;
          messages.scrollTop = messages.scrollHeight;
        } else if (payload.type === "tool_call") {
          cleanupLive();
          addStatusLine(`→ using tool: ${payload.tool}`);
        } else if (payload.type === "final") {
          cleanupLive();
          addBubble(payload.text, "assistant");
        } else if (payload.type === "error") {
          cleanupLive();
          addBubble(payload.text, "error");
        }
      }
    }
  }

  // Auto-grow the composer textarea with its content, up to the CSS
  // max-height (140px) -- past that, #message-input's own
  // `overflow-y: auto` takes over instead of the box (or the text)
  // silently getting cut off.
  function resizeInput() {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight}px`;
  }
  input.addEventListener("input", resizeInput);

  // Enter sends; Shift+Enter inserts a newline (textareas don't submit
  // their form on Enter by default, so this has to be done manually).
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    addBubble(message, "user");
    input.value = "";
    resizeInput();
    setBusy(true);

    try {
      await sendStreaming(message);
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
