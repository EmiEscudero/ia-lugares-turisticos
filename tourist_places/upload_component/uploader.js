/* Streamlit's v1 component messages; no remote assets or persistent storage. */
(() => {
  "use strict";
  const el = (id) => document.getElementById(id);
  let args, value = null, busy = false, sequence = 0;
  const send = (type, data) => window.parent.postMessage({isStreamlitMessage: true, type, ...data}, "*");
  const resize = () => send("streamlit:setFrameHeight", {height: el("drop").offsetHeight + 4});
  const notify = () => send("streamlit:setComponentValue", {value, dataType: "json"});
  function render() {
    if (!args) return;
    document.documentElement.lang = args.language;
    el("instruction").textContent = args.labels.drop_photo;
    el("help").textContent = args.labels.upload_help;
    el("choose").textContent = args.labels.choose_photo;
    el("remove").textContent = args.labels.remove_photo;
    el("choose").disabled = busy;
    el("remove").hidden = !value || busy;
    el("status").textContent = busy ? args.labels.uploading :
      value?.error ? args.labels[value.error] : value?.name || "";
    resize();
  }
  function clear() {
    sequence++; // Invalidate a previous FileReader completion.
    value = null; busy = false; el("photo").value = "";
    render(); notify();
  }
  function accept(file) {
    if (!file || !args) return;
    const current = ++sequence;
    const fail = (error) => {
      if (current !== sequence) return;
      busy = false; value = {error}; render(); notify();
    };
    if (!file.size || file.size > args.max_bytes) return fail("upload_size");
    if (!/\.(jpe?g|png|webp)$/i.test(file.name)) return fail("upload_format");
    busy = true; render();
    const reader = new FileReader();
    reader.onerror = () => fail("upload_unreadable");
    reader.onload = () => {
      if (current !== sequence) return;
      busy = false;
      value = {name: file.name, content: String(reader.result).split(",", 2)[1]};
      render(); notify();
    };
    reader.readAsDataURL(file);
  }
  el("choose").addEventListener("click", () => el("photo").click());
  el("photo").addEventListener("change", (event) => accept(event.target.files[0]));
  el("remove").addEventListener("click", clear);
  el("drop").addEventListener("dragover", (event) => {event.preventDefault(); el("drop").classList.add("over");});
  el("drop").addEventListener("dragleave", () => el("drop").classList.remove("over"));
  el("drop").addEventListener("drop", (event) => {
    event.preventDefault(); el("drop").classList.remove("over"); accept(event.dataTransfer.files[0]);
  });
  window.addEventListener("message", (event) => {
    if (event.source !== window.parent || event.data?.type !== "streamlit:render") return;
    args = event.data.args; render();
  });
  window.addEventListener("resize", resize);
  send("streamlit:componentReady", {apiVersion: 1});
})();
