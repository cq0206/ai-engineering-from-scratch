import { defineConfig, type HeadConfig } from "vitepress";
import sidebar from "./sidebar.json";

const SITE_URL = "https://cq0206.github.io/ai-engineering-from-scratch";
const SITE_TITLE = "AI 工程从零开始";
const SITE_DESC = "473 课时系统化 AI 工程课程 — 从数学基础到 Agent 生产部署";

export default defineConfig({
  title: SITE_TITLE,
  description: SITE_DESC,
  base: "/ai-engineering-from-scratch/",
  lang: "zh-CN",

  sitemap: {
    hostname: "https://cq0206.github.io",
    transformItems(items) {
      return items.map(item => ({
        ...item,
        url: `ai-engineering-from-scratch/${item.url}`,
      }));
    },
  },

  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: "/ai-engineering-from-scratch/logo.svg" }],
    ["meta", { property: "og:site_name", content: SITE_TITLE }],
    ["meta", { property: "og:locale", content: "zh_CN" }],
    ["meta", { name: "twitter:card", content: "summary" }],
  ],

  transformHead({ pageData }) {
    const head: HeadConfig[] = [];
    const title = pageData.title || SITE_TITLE;
    const desc = pageData.description || SITE_DESC;
    const url = `${SITE_URL}/${pageData.relativePath.replace(/\.md$/, ".html")}`;

    head.push(["meta", { property: "og:title", content: title }]);
    head.push(["meta", { property: "og:description", content: desc }]);
    head.push(["meta", { property: "og:url", content: url }]);
    head.push(["meta", { property: "og:type", content: "article" }]);
    head.push(["meta", { name: "twitter:title", content: title }]);
    head.push(["meta", { name: "twitter:description", content: desc }]);
    head.push(["link", { rel: "canonical", href: url }]);

    return head;
  },

  themeConfig: {
    nav: [
      { text: "首页", link: "/" },
      { text: "开始学习", link: "/phases/00-setup-and-tooling/" },
      { text: "← 返回 learn.traeai.com", link: "https://learn.traeai.com" },
    ],

    sidebar: {
      "/phases/": sidebar,
    },

    outline: {
      level: [2, 3],
      label: "目录",
    },

    search: {
      provider: "local",
      options: {
        translations: {
          button: { buttonText: "搜索课程" },
          modal: {
            noResultsText: "未找到结果",
            resetButtonTitle: "清除搜索",
            footer: { selectText: "选择", navigateText: "切换" },
          },
        },
      },
    },

    docFooter: {
      prev: "上一课",
      next: "下一课",
    },

    returnToTopLabel: "回到顶部",
    sidebarMenuLabel: "课程目录",
    darkModeSwitchLabel: "主题",

    socialLinks: [
      { icon: "github", link: "https://github.com/rohitg00/ai-engineering-from-scratch" },
    ],

    footer: {
      message: '基于 <a href="https://github.com/rohitg00/ai-engineering-from-scratch">AI Engineering from Scratch</a> (MIT License) 整理',
      copyright: "© 2025 learn.traeai.com",
    },

    editLink: {
      pattern: "https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/:path",
      text: "查看原文",
    },
  },

  markdown: {
    lineNumbers: true,
    math: true,
  },

  vue: {
    template: {
      compilerOptions: {
        // Allow custom HTML elements from source markdown
        isCustomElement: (tag) => tag.includes("-"),
      },
    },
  },

  vite: {
    build: {
      chunkSizeWarningLimit: 2000,
    },
  },
});
