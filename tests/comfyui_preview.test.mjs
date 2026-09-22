// CPU-only contract tests. No ComfyUI, browser, GPU or network dependencies.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../comfyui_client/web/h3_video.js", import.meta.url), "utf8");
const filename = `video_${"a".repeat(32)}.mp4`;
const payload = (name = filename) => ({ h3_videos: [{ filename: name, subfolder: "h3_api", type: "output" }] });

class Element {
    constructor(tag) {
        this.tag = tag;
        this.style = {};
        this.children = [];
        this.listeners = new Map();
        this.pauseCount = 0;
    }
    append(...elements) { this.children.push(...elements); }
    appendChild(element) { this.append(element); }
    addEventListener(name, callback) { this.listeners.set(name, callback); }
    pause() { this.pauseCount++; }
    load() { this.loaded = true; }
    removeAttribute(name) { delete this[name]; }
    remove() { this.removed = true; }
}

async function setup(name = "H3RemoteVideo") {
    let extension;
    const context = vm.createContext({
        URLSearchParams,
        document: { createElement: (tag) => new Element(tag) },
    });
    const appModule = new vm.SyntheticModule(["app"], function () {
        this.setExport("app", { registerExtension(value) { extension = value; } });
    }, { context });
    const apiModule = new vm.SyntheticModule(["api"], function () {
        // Exercise a ComfyUI reverse-proxy prefix as well as the output query.
        this.setExport("api", { apiURL: (url) => `/comfy${url}` });
    }, { context });
    const module = new vm.SourceTextModule(source, { context });
    await module.link((specifier) => {
        if (specifier === "../../../scripts/app.js") return appModule;
        if (specifier === "../../../scripts/api.js") return apiModule;
        throw new Error(`Unexpected import: ${specifier}`);
    });
    await module.evaluate();
    class Node {
        constructor() { this.size = [250, 400]; this.widgets = []; }
        onExecuted() { this.executed = (this.executed ?? 0) + 1; return "previous-hook"; }
        onRemoved() { this.removed = true; }
        addDOMWidget(name, type, element, options) {
            const widget = { name, type, element, options };
            this.widgets.push(widget);
            return widget;
        }
        computeSize() { return [340, 730]; }
        setSize(size) { this.size = size; }
        setDirtyCanvas() { this.dirty = true; }
    }
    await extension.beforeRegisterNodeDef(Node, { name });
    return new Node();
}

test("completed output shows native audio/video controls and same-origin download", async () => {
    const node = await setup();
    assert.equal(node.onExecuted(payload()), "previous-hook");
    assert.equal(node.widgets.length, 1);
    const widget = node.widgets[0];
    assert.equal(widget.options.serialize, false);
    assert.equal(widget.options.getMinHeight(), 300);
    assert.equal(widget.computeSize()[1], 300);
    const [video, links, status] = widget.element.children;
    assert.equal(video.tag, "video");
    assert.equal(video.controls, true);
    assert.equal(video.muted, false);
    assert.equal(video.autoplay, false);
    assert.equal(video.preload, "metadata");
    const url = new URL(video.src, "https://comfy.example");
    assert.equal(url.pathname, "/comfy/view");
    assert.equal(url.searchParams.get("filename"), filename);
    assert.equal(url.searchParams.get("subfolder"), "h3_api");
    assert.equal(url.searchParams.get("type"), "output");
    assert.equal([...url.searchParams].length, 3);
    const [download, open] = links.children;
    assert.equal(download.href, video.src);
    assert.equal(download.download, filename);
    assert.equal(open.href, video.src);
    assert.equal(open.target, "_blank");
    assert.equal(open.rel, "noopener noreferrer");
    assert.match(status.textContent, /下载/);
    assert.equal(node.dirty, true);
    let stopped = false;
    widget.element.listeners.get("pointerdown")({ stopPropagation() { stopped = true; } });
    assert.equal(stopped, true);
});

test("rerun reuses player; removing node releases media and preserves prior hooks", async () => {
    const node = await setup();
    node.onExecuted(payload());
    const element = node.widgets[0].element;
    const [video, links] = element.children;
    const next = `video_${"b".repeat(32)}.mp4`;
    node.onExecuted(payload(next));
    assert.equal(node.executed, 2);
    assert.equal(node.widgets.length, 1);
    assert.equal(links.children[0].download, next);
    assert.match(video.src, new RegExp(next));
    node.onRemoved();
    assert.equal(node.removed, true);
    assert.equal(element.removed, true);
    assert.equal(video.src, undefined);
    assert.equal(video.pauseCount, 3);
});

test("legacy/invalid payloads do not load files or arbitrary addresses", async () => {
    const node = await setup();
    for (const message of [
        {}, { text: ["/output/old.mp4"] }, payload("../../config.json"),
        payload("https://external.example/video.mp4"),
        { h3_videos: [{ filename, subfolder: "../", type: "output" }] },
        { h3_videos: [{ filename, subfolder: "h3_api", type: "input" }] },
    ]) node.onExecuted(message);
    assert.equal(node.widgets.length, 0);
    const other = await setup("UnrelatedNode");
    other.onExecuted(payload());
    assert.equal(other.widgets.length, 0);
});

test("playback error keeps download available and shows a useful message", async () => {
    const node = await setup();
    node.onExecuted(payload());
    const [video, links, status] = node.widgets[0].element.children;
    video.listeners.get("error")();
    assert.match(status.textContent, /预览失败/);
    assert.equal(links.children[0].download, filename);
});
