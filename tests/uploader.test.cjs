// Exercise browser events and Streamlit messages without opening a file dialog.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const source = fs.readFileSync(path.join(__dirname, "../tourist_places/upload_component/uploader.js"), "utf8");

function component() {
  const elements = Object.fromEntries(["drop", "instruction", "help", "choose", "remove", "photo", "status"].map(id => [id, {
    textContent: "", value: "", offsetHeight: 140, events: {},
    addEventListener(event, handler) {this.events[event] = handler;},
    classList: {add() {}, remove() {}}, click() {},
  }]));
  const messages = [], readers = [], events = {};
  const parent = {postMessage(message) {messages.push(message);}};
  const window = {parent, addEventListener(name, handler) {events[name] = handler;}};
  const document = {documentElement: {}, getElementById(id) {return elements[id];}};
  class FileReader {
    constructor() {readers.push(this);}
    readAsDataURL(file) {this.file = file;}
    finish() {this.result = "data:image/png;base64,AAECAw=="; this.onload();}
  }
  vm.runInNewContext(source, {window, document, FileReader});
  const render = language => events.message({source: parent, data: {type: "streamlit:render", args: {
    language, max_bytes: 100, labels: {choose_photo: language === "es" ? "Elegir una foto" : "Choose a photo", upload_size: "size", upload_format: "format", upload_unreadable: "unreadable"},
  }}});
  const upload = file => elements.photo.events.change({target: {files: [file]}});
  const values = () => messages.filter(m => m.type === "streamlit:setComponentValue").map(m => m.value);
  return {elements, messages, readers, render, upload, values, document};
}

test("language switch retains selected photo without sending it again", () => {
  const c = component();
  c.render("es"); c.upload({name: "México.png", size: 4}); c.readers[0].finish();
  assert.equal(c.values()[0].content, "AAECAw==");
  assert.equal(c.elements.status.textContent, "México.png");
  c.render("en");
  assert.equal(c.document.documentElement.lang, "en");
  assert.equal(c.elements.choose.textContent, "Choose a photo");
  assert.equal(c.elements.status.textContent, "México.png");
  assert.equal(c.values().length, 1);
});

test("oversized and unsupported files are rejected before reading", () => {
  const c = component(); c.render("es");
  c.upload({name: "large.jpg", size: 101});
  assert.equal(c.values().at(-1).error, "upload_size");
  c.upload({name: "animation.gif", size: 10});
  assert.equal(c.values().at(-1).error, "upload_format");
  assert.equal(c.readers.length, 0);
});

test("remove prevents a late reader from restoring the previous photo", () => {
  const c = component(); c.render("es");
  c.upload({name: "first.jpg", size: 10}); c.readers[0].finish();
  c.upload({name: "second.jpg", size: 10});
  c.elements.remove.events.click(); c.readers[1].finish();
  assert.equal(c.values().at(-1), null);
  assert.equal(c.elements.photo.value, "");
  assert.equal(c.elements.remove.hidden, true);
});

test("the newest drop wins if an earlier file finishes later", () => {
  const c = component(); c.render("en");
  c.upload({name: "first.jpg", size: 10});
  c.elements.drop.events.drop({preventDefault() {}, dataTransfer: {files: [{name: "second.png", size: 10}]}});
  c.readers[1].finish(); c.readers[0].finish();
  assert.equal(c.values().length, 1);
  assert.equal(c.values()[0].name, "second.png");
});
