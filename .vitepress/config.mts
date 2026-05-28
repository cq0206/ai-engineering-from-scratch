import { defineConfig } from "vitepress";
import sidebar from "./sidebar.json";

export default defineConfig({
  title: "AI 工程从零开始",
  description: "473 课时系统化 AI 工程课程 — 从数学基础到 Agent 生产部署",
  base: "/t/ai-engineering/",
  lang: "zh-CN",

  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: "/t/ai-engineering/logo.svg" }],
  ],

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
