import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const output = join(root, "site");
const config = JSON.parse(await readFile(join(root, "docs.json"), "utf8"));

function pageUrl(page) {
  return page === "introduction" ? "/" : `/${page}/`;
}

function titleFromPage(page) {
  return page.split("/").at(-1).replace(/-/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function pagesFromNavigation() {
  return config.navigation.tabs.flatMap((tab) =>
    tab.groups.flatMap((group) => (group.pages ?? []).map((page) => ({ ...group, page }))),
  );
}

function parseDocument(source) {
  const match = source.match(/^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/);
  if (!match) return { title: "IdentArk", description: "", body: source };
  const title = match[1].match(/^title:\s*["']?(.*?)["']?\s*$/m)?.[1] ?? "IdentArk";
  const description = match[1].match(/^description:\s*["']?(.*?)["']?\s*$/m)?.[1] ?? "";
  return { title, description, body: match[2] };
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[character]);
}

function transformMintlifyMdx(source) {
  return source
    .replace(/<CodeGroup[^>]*>/g, "")
    .replace(/<\/CodeGroup>/g, "")
    .replace(/<CardGroup[^>]*>/g, '<div class="card-grid">')
    .replace(/<\/CardGroup>/g, "</div>")
    .replace(/<Card\s+title="([^"]+)"[^>]*href="([^"]+)"[^>]*>/g, '<a class="card" href="$2"><h3>$1</h3>')
    .replace(/<Card\s+title="([^"]+)"[^>]*>/g, '<section class="card"><h3>$1</h3>')
    .replace(/<\/Card>/g, "</a>")
    .replace(/<(Note|Tip|Info|Warning)>/g, (_, kind) => `<aside class="callout ${kind.toLowerCase()}"><strong>${kind}</strong>`)
    .replace(/<\/(Note|Tip|Info|Warning)>/g, "</aside>")
    .replace(/<Steps>/g, '<div class="steps">')
    .replace(/<\/Steps>/g, "</div>")
    .replace(/<Step\s+title="([^"]+)"[^>]*>/g, '<section class="step"><h3>$1</h3>')
    .replace(/<\/Step>/g, "</section>")
    .replace(/<ParamField\s+([^>]*)\/>/g, (_, attributes) => {
      const name = attributes.match(/name="([^"]+)"/)?.[1] ?? "Parameter";
      const type = attributes.match(/type="([^"]+)"/)?.[1];
      return `<div class="parameter"><code>${name}</code>${type ? ` <span>${type}</span>` : ""}</div>`;
    });
}

function sidebar(activePage) {
  return config.navigation.tabs.map((tab) => `
    <section class="nav-tab"><p>${escapeHtml(tab.tab)}</p>
      ${tab.groups.map((group) => `<div class="nav-group"><h2>${escapeHtml(group.group)}</h2>${(group.pages ?? []).map((page) => `<a class="${page === activePage ? "active" : ""}" href="${pageUrl(page)}">${escapeHtml(titleFromPage(page))}</a>`).join("")}</div>`).join("")}
    </section>`).join("");
}

function layout({ page, title, description, content }) {
  const navbar = (config.navbar.links ?? []).map((link) => `<a href="${link.href}">${escapeHtml(link.label)}</a>`).join("");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="${escapeHtml(description)}"><title>${escapeHtml(title)} | IdentArk</title>
<link rel="icon" href="/favicon.svg"><style>
:root{--ink:#102a43;--muted:#58708a;--line:#dbe5ed;--brand:#0b8a7e;--wash:#f6fbfa}*{box-sizing:border-box}body{margin:0;color:var(--ink);font:16px/1.65 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}a{color:#067b71;text-decoration:none}a:hover{text-decoration:underline}.top{height:64px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 32px;position:sticky;top:0;background:#fff;z-index:2}.top b{font-size:20px}.top nav{display:flex;gap:20px;font-size:14px}.shell{display:grid;grid-template-columns:272px minmax(0,760px);max-width:1180px;margin:auto}.sidebar{border-right:1px solid var(--line);min-height:calc(100vh - 64px);padding:26px 22px}.nav-tab>p{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700;margin:18px 0 8px}.nav-group{margin:0 0 20px}.nav-group h2{font-size:13px;color:var(--muted);margin:0 0 5px}.nav-group a{display:block;color:var(--ink);padding:4px 9px;border-radius:6px;font-size:14px}.nav-group a.active{background:#e7f6f3;color:#056d64;font-weight:700}.content{padding:52px 56px 80px;min-width:0}.content h1{font-size:42px;line-height:1.12;letter-spacing:-.03em;margin:0 0 16px}.content h2{font-size:26px;line-height:1.25;margin:46px 0 14px}.content h3{font-size:18px;margin:20px 0 8px}.content pre{background:#0d1b2a;color:#e8f2f5;padding:18px;border-radius:10px;overflow:auto}.content code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.content :not(pre)>code{background:#eef4f7;padding:2px 5px;border-radius:4px}.callout{border-left:4px solid var(--brand);background:var(--wash);padding:14px 17px;margin:22px 0;border-radius:0 8px 8px 0}.callout.warning{border-color:#ca8a04;background:#fffbeb}.steps{border-left:2px solid #b6ddd8;margin:22px 0;padding-left:24px}.step{margin:0 0 28px}.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:22px 0}.card{display:block;border:1px solid var(--line);border-radius:10px;padding:16px;color:var(--ink)}.card:hover{text-decoration:none;border-color:var(--brand)}.card h3{margin-top:0}.parameter{border:1px solid var(--line);padding:10px;border-radius:6px}.parameter span{color:var(--muted)}@media(max-width:780px){.shell{display:block}.sidebar{display:none}.content{padding:32px 22px}.top{padding:0 18px}.top nav{gap:10px}.top nav a:not(:last-child){display:none}.content h1{font-size:34px}}</style></head>
<body><header class="top"><a href="/"><b>IdentArk</b></a><nav>${navbar}<a href="${config.navbar.primary.href}">${escapeHtml(config.navbar.primary.label)}</a></nav></header>
<div class="shell"><aside class="sidebar">${sidebar(page)}</aside><main class="content">${content}</main></div></body></html>`;
}

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(join(root, "images"), join(output, "images"), { recursive: true });
await cp(join(root, "favicon.svg"), join(output, "favicon.svg"));

for (const { page } of pagesFromNavigation()) {
  const source = await readFile(join(root, `${page}.mdx`), "utf8");
  const document = parseDocument(source);
  const html = marked.parse(transformMintlifyMdx(document.body));
  const destination = page === "introduction" ? join(output, "index.html") : join(output, page, "index.html");
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, layout({ page, ...document, content: html }));
}

const spec = JSON.parse(await readFile(join(root, "api-reference", "openapi.json"), "utf8"));
const endpoints = Object.entries(spec.paths ?? {}).flatMap(([path, methods]) => Object.keys(methods).map((method) => `<li><code>${method.toUpperCase()}</code> ${escapeHtml(path)}</li>`)).join("");
const apiDestination = join(output, "api-reference", "endpoints", "index.html");
await mkdir(dirname(apiDestination), { recursive: true });
await writeFile(apiDestination, layout({ page: "api-reference/introduction", title: "API endpoints", description: "IdentArk public API endpoints.", content: `<h1>API endpoints</h1><p>Generated from the public IdentArk OpenAPI contract.</p><ul>${endpoints}</ul>` }));

console.log(`Built ${pagesFromNavigation().length} documentation pages in ${output}`);
