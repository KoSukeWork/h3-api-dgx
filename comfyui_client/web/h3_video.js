import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Only ComfyUI's local output endpoint is exposed to the browser, never the DGX key/URL.
function videoURL(file) {
    if (!file || !/^video_[a-f0-9]{32}\.mp4$/.test(file.filename)
        || file.subfolder !== "h3_api" || file.type !== "output") return null;
    const query = new URLSearchParams({
        filename: file.filename, subfolder: "h3_api", type: "output",
    });
    return api.apiURL(`/view?${query}`);
}

function createPreview(node) {
    const container = document.createElement("div");
    Object.assign(container.style, {
        display: "flex", flexDirection: "column", gap: "6px", padding: "6px",
        boxSizing: "border-box", width: "100%", height: "100%", background: "#181818",
    });
    // Prevent LiteGraph dragging/shortcuts from swallowing native media controls.
    for (const event of ["pointerdown", "mousedown", "click", "dblclick", "keydown"]) {
        container.addEventListener(event, (e) => e.stopPropagation());
    }
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.autoplay = false;
    video.muted = false;
    Object.assign(video.style, { width: "100%", flex: "1", minHeight: "0", objectFit: "contain" });
    const links = document.createElement("div");
    Object.assign(links.style, { display: "flex", gap: "16px", flexShrink: "0" });
    const download = document.createElement("a");
    download.textContent = "下载 MP4";
    const open = document.createElement("a");
    open.textContent = "新窗口播放";
    open.target = "_blank";
    open.rel = "noopener noreferrer";
    for (const link of [download, open]) {
        link.style.color = "#8ecfff";
        links.appendChild(link);
    }
    const status = document.createElement("div");
    Object.assign(status.style, { color: "#ddd", fontSize: "12px", flexShrink: "0" });
    video.addEventListener("error", () => {
        status.textContent = "预览失败：可尝试下载后播放；若下载也失败，请检查输出文件是否仍存在。";
    });
    container.append(video, links, status);
    const widget = node.addDOMWidget("h3_video_preview", "h3_video", container, {
        serialize: false, getMinHeight: () => 300, getHeight: () => 300,
    });
    widget.computeSize = () => [320, 300];
    widget.options.serialize = false;
    return {
        show(file, url) {
            video.pause();
            video.src = url;
            download.href = url;
            download.download = file.filename;
            open.href = url;
            status.textContent = "点击播放可听到音频；下载保存到浏览器所在电脑。";
            video.load();
        },
        dispose() {
            video.pause();
            video.removeAttribute("src");
            video.load();
            container.remove();
        },
    };
}

app.registerExtension({
    name: "H3.RemoteVideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3RemoteVideo") return;
        const previews = new WeakMap();
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const file = message?.h3_videos?.[0];
            const url = videoURL(file);
            if (!url) return result;
            let preview = previews.get(this);
            if (!preview) {
                preview = createPreview(this);
                previews.set(this, preview);
                const size = this.computeSize();
                this.setSize([Math.max(this.size[0], size[0], 340), Math.max(this.size[1], size[1])]);
            }
            preview.show(file, url);
            this.setDirtyCanvas(true, true);
            return result;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            previews.get(this)?.dispose();
            previews.delete(this);
            return onRemoved?.apply(this, arguments);
        };
    },
});
