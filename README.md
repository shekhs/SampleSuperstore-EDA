<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>EDA on SuperStore Sales Dataset </title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>




<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>



<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/*
 * Webkit scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-corner {
  background: var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-thumb {
  background: rgb(var(--jp-scrollbar-thumb-color));
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-right: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

[data-jp-theme-scrollbars='true'] ::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
  border-bottom: var(--jp-scrollbar-endpad) solid
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar */

[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar::-webkit-scrollbar,
[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-corner,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-corner {
  background-color: transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-thumb,
[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
  border: var(--jp-scrollbar-thumb-margin) solid transparent;
  background-clip: content-box;
  border-radius: var(--jp-scrollbar-thumb-radius);
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-hscrollbar::-webkit-scrollbar-track:horizontal {
  border-left: var(--jp-scrollbar-endpad) solid transparent;
  border-right: var(--jp-scrollbar-endpad) solid transparent;
}

[data-jp-theme-scrollbars='true']
  .CodeMirror-vscrollbar::-webkit-scrollbar-track:vertical {
  border-top: var(--jp-scrollbar-endpad) solid transparent;
  border-bottom: var(--jp-scrollbar-endpad) solid transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0px solid transparent;
  border-right: 0px solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0px solid transparent;
  border-bottom: 0px solid transparent;
}

/*
 * Phosphor
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Widget, /* </DEPRECATED> */
.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  cursor: default;
}


/* <DEPRECATED> */ .p-Widget.p-mod-hidden, /* </DEPRECATED> */
.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-CommandPalette, /* </DEPRECATED> */
.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-CommandPalette-search, /* </DEPRECATED> */
.lm-CommandPalette-search {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-content, /* </DEPRECATED> */
.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-CommandPalette-header, /* </DEPRECATED> */
.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}


/* <DEPRECATED> */ .p-CommandPalette-item, /* </DEPRECATED> */
.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}


/* <DEPRECATED> */ .p-CommandPalette-itemIcon, /* </DEPRECATED> */
.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemContent, /* </DEPRECATED> */
.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}


/* <DEPRECATED> */ .p-CommandPalette-itemShortcut, /* </DEPRECATED> */
.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-CommandPalette-itemLabel, /* </DEPRECATED> */
.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
	border:1px solid transparent;
  background-color: transparent;
  position: absolute;
	z-index:1;
	right:3%;
	top: 0;
	bottom: 0;
	margin: auto;
	padding: 7px 0;
	display: none;
	vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
	content: "X";
	display: block;
	width: 15px;
	height: 15px;
	text-align: center;
	color:#000;
	font-weight: normal;
	font-size: 12px;
	cursor: pointer;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-DockPanel, /* </DEPRECATED> */
.lm-DockPanel {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-widget, /* </DEPRECATED> */
.lm-DockPanel-widget {
  z-index: 0;
}


/* <DEPRECATED> */ .p-DockPanel-tabBar, /* </DEPRECATED> */
.lm-DockPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-DockPanel-handle, /* </DEPRECATED> */
.lm-DockPanel-handle {
  z-index: 2;
}


/* <DEPRECATED> */ .p-DockPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-DockPanel-handle:after, /* </DEPRECATED> */
.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='horizontal']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-DockPanel-handle[data-orientation='vertical']:after,
/* </DEPRECATED> */
.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}


/* <DEPRECATED> */ .p-DockPanel-overlay, /* </DEPRECATED> */
.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}


/* <DEPRECATED> */ .p-DockPanel-overlay.p-mod-hidden, /* </DEPRECATED> */
.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-Menu, /* </DEPRECATED> */
.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-Menu-content, /* </DEPRECATED> */
.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}


/* <DEPRECATED> */ .p-Menu-item, /* </DEPRECATED> */
.lm-Menu-item {
  display: table-row;
}


/* <DEPRECATED> */
.p-Menu-item.p-mod-hidden,
.p-Menu-item.p-mod-collapsed,
/* </DEPRECATED> */
.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}


/* <DEPRECATED> */
.p-Menu-itemIcon,
.p-Menu-itemSubmenuIcon,
/* </DEPRECATED> */
.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}


/* <DEPRECATED> */ .p-Menu-itemLabel, /* </DEPRECATED> */
.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}


/* <DEPRECATED> */ .p-Menu-itemShortcut, /* </DEPRECATED> */
.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-MenuBar, /* </DEPRECATED> */
.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-MenuBar-content, /* </DEPRECATED> */
.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}


/* <DEPRECATED> */ .p--MenuBar-item, /* </DEPRECATED> */
.lm-MenuBar-item {
  box-sizing: border-box;
}


/* <DEPRECATED> */
.p-MenuBar-itemIcon,
.p-MenuBar-itemLabel,
/* </DEPRECATED> */
.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-ScrollBar, /* </DEPRECATED> */
.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='horizontal'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-ScrollBar[data-orientation='vertical'],
/* </DEPRECATED> */
.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-ScrollBar-button, /* </DEPRECATED> */
.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-track, /* </DEPRECATED> */
.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}


/* <DEPRECATED> */ .p-ScrollBar-thumb, /* </DEPRECATED> */
.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-SplitPanel-child, /* </DEPRECATED> */
.lm-SplitPanel-child {
  z-index: 0;
}


/* <DEPRECATED> */ .p-SplitPanel-handle, /* </DEPRECATED> */
.lm-SplitPanel-handle {
  z-index: 1;
}


/* <DEPRECATED> */ .p-SplitPanel-handle.p-mod-hidden, /* </DEPRECATED> */
.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-SplitPanel-handle:after, /* </DEPRECATED> */
.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}


/* <DEPRECATED> */
.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after,
/* </DEPRECATED> */
.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabBar, /* </DEPRECATED> */
.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='horizontal'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}


/* <DEPRECATED> */ .p-TabBar[data-orientation='vertical'], /* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}


/* <DEPRECATED> */ .p-TabBar-content, /* </DEPRECATED> */
.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}


/* <DEPRECATED> */
.p-TabBar[data-orientation='vertical'] > .p-TabBar-content,
/* </DEPRECATED> */
.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}


/* <DEPRECATED> */ .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


/* <DEPRECATED> */
.p-TabBar-tabIcon,
.p-TabBar-tabCloseIcon,
/* </DEPRECATED> */
.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


/* <DEPRECATED> */ .p-TabBar-tabLabel, /* </DEPRECATED> */
.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
}


/* <DEPRECATED> */ .p-TabBar-tab.p-mod-hidden, /* </DEPRECATED> */
.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}


.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}


/* <DEPRECATED> */ .p-TabBar.p-mod-dragging .p-TabBar-tab, /* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


/* <DEPRECATED> */
.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging,
/* </DEPRECATED> */
.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing : border-box;
  background: inherit;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ .p-TabPanel-tabBar, /* </DEPRECATED> */
.lm-TabPanel-tabBar {
  z-index: 1;
}


/* <DEPRECATED> */ .p-TabPanel-stackedPanel, /* </DEPRECATED> */
.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

@charset "UTF-8";
html{
  -webkit-box-sizing:border-box;
          box-sizing:border-box; }

*,
*::before,
*::after{
  -webkit-box-sizing:inherit;
          box-sizing:inherit; }

body{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none;
  color:#182026;
  font-family:-apple-system, "BlinkMacSystemFont", "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Open Sans", "Helvetica Neue", "Icons16", sans-serif; }

p{
  margin-bottom:10px;
  margin-top:0; }

small{
  font-size:12px; }

strong{
  font-weight:600; }

::-moz-selection{
  background:rgba(125, 188, 255, 0.6); }

::selection{
  background:rgba(125, 188, 255, 0.6); }
.bp3-heading{
  color:#182026;
  font-weight:600;
  margin:0 0 10px;
  padding:0; }
  .bp3-dark .bp3-heading{
    color:#f5f8fa; }

h1.bp3-heading, .bp3-running-text h1{
  font-size:36px;
  line-height:40px; }

h2.bp3-heading, .bp3-running-text h2{
  font-size:28px;
  line-height:32px; }

h3.bp3-heading, .bp3-running-text h3{
  font-size:22px;
  line-height:25px; }

h4.bp3-heading, .bp3-running-text h4{
  font-size:18px;
  line-height:21px; }

h5.bp3-heading, .bp3-running-text h5{
  font-size:16px;
  line-height:19px; }

h6.bp3-heading, .bp3-running-text h6{
  font-size:14px;
  line-height:16px; }
.bp3-ui-text{
  font-size:14px;
  font-weight:400;
  letter-spacing:0;
  line-height:1.28581;
  text-transform:none; }

.bp3-monospace-text{
  font-family:monospace;
  text-transform:none; }

.bp3-text-muted{
  color:#5c7080; }
  .bp3-dark .bp3-text-muted{
    color:#a7b6c2; }

.bp3-text-disabled{
  color:rgba(92, 112, 128, 0.6); }
  .bp3-dark .bp3-text-disabled{
    color:rgba(167, 182, 194, 0.6); }

.bp3-text-overflow-ellipsis{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal; }
.bp3-running-text{
  font-size:14px;
  line-height:1.5; }
  .bp3-running-text h1{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h1{
      color:#f5f8fa; }
  .bp3-running-text h2{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h2{
      color:#f5f8fa; }
  .bp3-running-text h3{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h3{
      color:#f5f8fa; }
  .bp3-running-text h4{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h4{
      color:#f5f8fa; }
  .bp3-running-text h5{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h5{
      color:#f5f8fa; }
  .bp3-running-text h6{
    color:#182026;
    font-weight:600;
    margin-bottom:20px;
    margin-top:40px; }
    .bp3-dark .bp3-running-text h6{
      color:#f5f8fa; }
  .bp3-running-text hr{
    border:none;
    border-bottom:1px solid rgba(16, 22, 26, 0.15);
    margin:20px 0; }
    .bp3-dark .bp3-running-text hr{
      border-color:rgba(255, 255, 255, 0.15); }
  .bp3-running-text p{
    margin:0 0 10px;
    padding:0; }

.bp3-text-large{
  font-size:16px; }

.bp3-text-small{
  font-size:12px; }
a{
  color:#106ba3;
  text-decoration:none; }
  a:hover{
    color:#106ba3;
    cursor:pointer;
    text-decoration:underline; }
  a .bp3-icon, a .bp3-icon-standard, a .bp3-icon-large{
    color:inherit; }
  a code,
  .bp3-dark a code{
    color:inherit; }
  .bp3-dark a,
  .bp3-dark a:hover{
    color:#48aff0; }
    .bp3-dark a .bp3-icon, .bp3-dark a .bp3-icon-standard, .bp3-dark a .bp3-icon-large,
    .bp3-dark a:hover .bp3-icon,
    .bp3-dark a:hover .bp3-icon-standard,
    .bp3-dark a:hover .bp3-icon-large{
      color:inherit; }
.bp3-running-text code, .bp3-code{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  font-size:smaller;
  padding:2px 5px; }
  .bp3-dark .bp3-running-text code, .bp3-running-text .bp3-dark code, .bp3-dark .bp3-code{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
  .bp3-running-text a > code, a > .bp3-code{
    color:#137cbd; }
    .bp3-dark .bp3-running-text a > code, .bp3-running-text .bp3-dark a > code, .bp3-dark a > .bp3-code{
      color:inherit; }

.bp3-running-text pre, .bp3-code-block{
  font-family:monospace;
  text-transform:none;
  background:rgba(255, 255, 255, 0.7);
  border-radius:3px;
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
  color:#182026;
  display:block;
  font-size:13px;
  line-height:1.4;
  margin:10px 0;
  padding:13px 15px 12px;
  word-break:break-all;
  word-wrap:break-word; }
  .bp3-dark .bp3-running-text pre, .bp3-running-text .bp3-dark pre, .bp3-dark .bp3-code-block{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
  .bp3-running-text pre > code, .bp3-code-block > code{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit;
    font-size:inherit;
    padding:0; }

.bp3-running-text kbd, .bp3-key{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#5c7080;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-family:inherit;
  font-size:12px;
  height:24px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  line-height:24px;
  min-width:24px;
  padding:3px 6px;
  vertical-align:middle; }
  .bp3-running-text kbd .bp3-icon, .bp3-key .bp3-icon, .bp3-running-text kbd .bp3-icon-standard, .bp3-key .bp3-icon-standard, .bp3-running-text kbd .bp3-icon-large, .bp3-key .bp3-icon-large{
    margin-right:5px; }
  .bp3-dark .bp3-running-text kbd, .bp3-running-text .bp3-dark kbd, .bp3-dark .bp3-key{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#a7b6c2; }
.bp3-running-text blockquote, .bp3-blockquote{
  border-left:solid 4px rgba(167, 182, 194, 0.5);
  margin:0 0 10px;
  padding:0 20px; }
  .bp3-dark .bp3-running-text blockquote, .bp3-running-text .bp3-dark blockquote, .bp3-dark .bp3-blockquote{
    border-color:rgba(115, 134, 148, 0.5); }
.bp3-running-text ul,
.bp3-running-text ol, .bp3-list{
  margin:10px 0;
  padding-left:30px; }
  .bp3-running-text ul li:not(:last-child), .bp3-running-text ol li:not(:last-child), .bp3-list li:not(:last-child){
    margin-bottom:5px; }
  .bp3-running-text ul ol, .bp3-running-text ol ol, .bp3-list ol,
  .bp3-running-text ul ul,
  .bp3-running-text ol ul,
  .bp3-list ul{
    margin-top:5px; }

.bp3-list-unstyled{
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-list-unstyled li{
    padding:0; }
.bp3-rtl{
  text-align:right; }

.bp3-dark{
  color:#f5f8fa; }

:focus{
  outline:rgba(19, 124, 189, 0.6) auto 2px;
  outline-offset:2px;
  -moz-outline-radius:6px; }

.bp3-focus-disabled :focus{
  outline:none !important; }
  .bp3-focus-disabled :focus ~ .bp3-control-indicator{
    outline:none !important; }

.bp3-alert{
  max-width:400px;
  padding:20px; }

.bp3-alert-body{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-alert-body .bp3-icon{
    font-size:40px;
    margin-right:20px;
    margin-top:0; }

.bp3-alert-contents{
  word-break:break-word; }

.bp3-alert-footer{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:reverse;
      -ms-flex-direction:row-reverse;
          flex-direction:row-reverse;
  margin-top:10px; }
  .bp3-alert-footer .bp3-button{
    margin-left:10px; }
.bp3-breadcrumbs{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  cursor:default;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:wrap;
      flex-wrap:wrap;
  height:30px;
  list-style:none;
  margin:0;
  padding:0; }
  .bp3-breadcrumbs > li{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }
    .bp3-breadcrumbs > li::after{
      background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M10.71 7.29l-4-4a1.003 1.003 0 00-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 001.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z' fill='%235C7080'/%3e%3c/svg%3e");
      content:"";
      display:block;
      height:16px;
      margin:0 5px;
      width:16px; }
    .bp3-breadcrumbs > li:last-of-type::after{
      display:none; }

.bp3-breadcrumb,
.bp3-breadcrumb-current,
.bp3-breadcrumbs-collapsed{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  font-size:16px; }

.bp3-breadcrumb,
.bp3-breadcrumbs-collapsed{
  color:#5c7080; }

.bp3-breadcrumb:hover{
  text-decoration:none; }

.bp3-breadcrumb.bp3-disabled{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-breadcrumb .bp3-icon{
  margin-right:5px; }

.bp3-breadcrumb-current{
  color:inherit;
  font-weight:600; }
  .bp3-breadcrumb-current .bp3-input{
    font-size:inherit;
    font-weight:inherit;
    vertical-align:baseline; }

.bp3-breadcrumbs-collapsed{
  background:#ced9e0;
  border:none;
  border-radius:3px;
  cursor:pointer;
  margin-right:2px;
  padding:1px 5px;
  vertical-align:text-bottom; }
  .bp3-breadcrumbs-collapsed::before{
    background:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cg fill='%235C7080'%3e%3ccircle cx='2' cy='8.03' r='2'/%3e%3ccircle cx='14' cy='8.03' r='2'/%3e%3ccircle cx='8' cy='8.03' r='2'/%3e%3c/g%3e%3c/svg%3e") center no-repeat;
    content:"";
    display:block;
    height:16px;
    width:16px; }
  .bp3-breadcrumbs-collapsed:hover{
    background:#bfccd6;
    color:#182026;
    text-decoration:none; }

.bp3-dark .bp3-breadcrumb,
.bp3-dark .bp3-breadcrumbs-collapsed{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumbs > li::after{
  color:#a7b6c2; }

.bp3-dark .bp3-breadcrumb.bp3-disabled{
  color:rgba(167, 182, 194, 0.6); }

.bp3-dark .bp3-breadcrumb-current{
  color:#f5f8fa; }

.bp3-dark .bp3-breadcrumbs-collapsed{
  background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-breadcrumbs-collapsed:hover{
    background:rgba(16, 22, 26, 0.6);
    color:#f5f8fa; }
.bp3-button{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  min-height:30px;
  min-width:30px; }
  .bp3-button > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-button > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-button::before,
  .bp3-button > *{
    margin-right:7px; }
  .bp3-button:empty::before,
  .bp3-button > :last-child{
    margin-right:0; }
  .bp3-button:empty{
    padding:0 !important; }
  .bp3-button:disabled, .bp3-button.bp3-disabled{
    cursor:not-allowed; }
  .bp3-button.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button.bp3-align-right,
  .bp3-align-right .bp3-button{
    text-align:right; }
  .bp3-button.bp3-align-left,
  .bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-button:not([class*="bp3-intent-"]){
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026; }
    .bp3-button:not([class*="bp3-intent-"]):hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-button:not([class*="bp3-intent-"]):active, .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active:hover, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active, .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-button.bp3-intent-primary{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover, .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-primary:hover{
      background-color:#106ba3;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:active, .bp3-button.bp3-intent-primary.bp3-active{
      background-color:#0e5a8a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-primary:disabled, .bp3-button.bp3-intent-primary.bp3-disabled{
      background-color:rgba(19, 124, 189, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-success{
    background-color:#0f9960;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-success:hover, .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-success:hover{
      background-color:#0d8050;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:active, .bp3-button.bp3-intent-success.bp3-active{
      background-color:#0a6640;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-success:disabled, .bp3-button.bp3-intent-success.bp3-disabled{
      background-color:rgba(15, 153, 96, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-warning{
    background-color:#d9822b;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover, .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-warning:hover{
      background-color:#bf7326;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:active, .bp3-button.bp3-intent-warning.bp3-active{
      background-color:#a66321;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-warning:disabled, .bp3-button.bp3-intent-warning.bp3-disabled{
      background-color:rgba(217, 130, 43, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button.bp3-intent-danger{
    background-color:#db3737;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover, .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      color:#ffffff; }
    .bp3-button.bp3-intent-danger:hover{
      background-color:#c23030;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:active, .bp3-button.bp3-intent-danger.bp3-active{
      background-color:#a82a2a;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-button.bp3-intent-danger:disabled, .bp3-button.bp3-intent-danger.bp3-disabled{
      background-color:rgba(219, 55, 55, 0.5);
      background-image:none;
      border-color:transparent;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.6); }
  .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
    stroke:#ffffff; }
  .bp3-button.bp3-large,
  .bp3-large .bp3-button{
    min-height:40px;
    min-width:40px;
    font-size:16px;
    padding:5px 15px; }
    .bp3-button.bp3-large::before,
    .bp3-button.bp3-large > *,
    .bp3-large .bp3-button::before,
    .bp3-large .bp3-button > *{
      margin-right:10px; }
    .bp3-button.bp3-large:empty::before,
    .bp3-button.bp3-large > :last-child,
    .bp3-large .bp3-button:empty::before,
    .bp3-large .bp3-button > :last-child{
      margin-right:0; }
  .bp3-button.bp3-small,
  .bp3-small .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-button.bp3-loading{
    position:relative; }
    .bp3-button.bp3-loading[class*="bp3-icon-"]::before{
      visibility:hidden; }
    .bp3-button.bp3-loading .bp3-button-spinner{
      margin:0;
      position:absolute; }
    .bp3-button.bp3-loading > :not(.bp3-button-spinner){
      visibility:hidden; }
  .bp3-button[class*="bp3-icon-"]::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    color:#5c7080; }
  .bp3-button .bp3-icon, .bp3-button .bp3-icon-standard, .bp3-button .bp3-icon-large{
    color:#5c7080; }
    .bp3-button .bp3-icon.bp3-align-right, .bp3-button .bp3-icon-standard.bp3-align-right, .bp3-button .bp3-icon-large.bp3-align-right{
      margin-left:7px; }
  .bp3-button .bp3-icon:first-child:last-child,
  .bp3-button .bp3-spinner + .bp3-icon:last-child{
    margin:0 -7px; }
  .bp3-dark .bp3-button:not([class*="bp3-intent-"]){
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover, .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-button:not([class*="bp3-intent-"]):disabled.bp3-active, .bp3-dark .bp3-button:not([class*="bp3-intent-"]).bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"])[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
    .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-button:not([class*="bp3-intent-"]) .bp3-icon-large{
      color:#a7b6c2; }
  .bp3-dark .bp3-button[class*="bp3-intent-"]{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:hover{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:active, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-active{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-button[class*="bp3-intent-"]:disabled, .bp3-dark .bp3-button[class*="bp3-intent-"].bp3-disabled{
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(255, 255, 255, 0.3); }
    .bp3-dark .bp3-button[class*="bp3-intent-"] .bp3-button-spinner .bp3-spinner-head{
      stroke:#8a9ba8; }
  .bp3-button:disabled::before,
  .bp3-button:disabled .bp3-icon, .bp3-button:disabled .bp3-icon-standard, .bp3-button:disabled .bp3-icon-large, .bp3-button.bp3-disabled::before,
  .bp3-button.bp3-disabled .bp3-icon, .bp3-button.bp3-disabled .bp3-icon-standard, .bp3-button.bp3-disabled .bp3-icon-large, .bp3-button[class*="bp3-intent-"]::before,
  .bp3-button[class*="bp3-intent-"] .bp3-icon, .bp3-button[class*="bp3-intent-"] .bp3-icon-standard, .bp3-button[class*="bp3-intent-"] .bp3-icon-large{
    color:inherit !important; }
  .bp3-button.bp3-minimal{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button.bp3-minimal:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-minimal:active, .bp3-button.bp3-minimal.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-minimal:disabled, .bp3-button.bp3-minimal:disabled:hover, .bp3-button.bp3-minimal.bp3-disabled, .bp3-button.bp3-minimal.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-minimal{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-minimal:hover, .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-minimal:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-minimal:active, .bp3-dark .bp3-button.bp3-minimal.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-minimal:disabled, .bp3-dark .bp3-button.bp3-minimal:disabled:hover, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-minimal:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover, .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover, .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover, .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-minimal.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover, .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-minimal.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-minimal.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button.bp3-outlined{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    border:1px solid rgba(24, 32, 38, 0.2);
    -webkit-box-sizing:border-box;
            box-sizing:border-box; }
    .bp3-button.bp3-outlined:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button.bp3-outlined:active, .bp3-button.bp3-outlined.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button.bp3-outlined{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button.bp3-outlined:hover, .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button.bp3-outlined:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button.bp3-outlined:active, .bp3-dark .bp3-button.bp3-outlined.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button.bp3-outlined:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined:disabled:hover.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover, .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover, .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover, .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover, .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button.bp3-outlined.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
    .bp3-button.bp3-outlined:disabled, .bp3-button.bp3-outlined.bp3-disabled, .bp3-button.bp3-outlined:disabled:hover, .bp3-button.bp3-outlined.bp3-disabled:hover{
      border-color:rgba(92, 112, 128, 0.1); }
    .bp3-dark .bp3-button.bp3-outlined{
      border-color:rgba(255, 255, 255, 0.4); }
      .bp3-dark .bp3-button.bp3-outlined:disabled, .bp3-dark .bp3-button.bp3-outlined:disabled:hover, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-disabled:hover{
        border-color:rgba(255, 255, 255, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-primary{
      border-color:rgba(16, 107, 163, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
        border-color:rgba(16, 107, 163, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary{
        border-color:rgba(72, 175, 240, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-primary.bp3-disabled{
          border-color:rgba(72, 175, 240, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-success{
      border-color:rgba(13, 128, 80, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
        border-color:rgba(13, 128, 80, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success{
        border-color:rgba(61, 204, 145, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-success.bp3-disabled{
          border-color:rgba(61, 204, 145, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-warning{
      border-color:rgba(191, 115, 38, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
        border-color:rgba(191, 115, 38, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning{
        border-color:rgba(255, 179, 102, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-warning.bp3-disabled{
          border-color:rgba(255, 179, 102, 0.2); }
    .bp3-button.bp3-outlined.bp3-intent-danger{
      border-color:rgba(194, 48, 48, 0.6); }
      .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
        border-color:rgba(194, 48, 48, 0.2); }
      .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger{
        border-color:rgba(255, 115, 115, 0.6); }
        .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger:disabled, .bp3-dark .bp3-button.bp3-outlined.bp3-intent-danger.bp3-disabled{
          border-color:rgba(255, 115, 115, 0.2); }

a.bp3-button{
  text-align:center;
  text-decoration:none;
  -webkit-transition:none;
  transition:none; }
  a.bp3-button, a.bp3-button:hover, a.bp3-button:active{
    color:#182026; }
  a.bp3-button.bp3-disabled{
    color:rgba(92, 112, 128, 0.6); }

.bp3-button-text{
  -webkit-box-flex:0;
      -ms-flex:0 1 auto;
          flex:0 1 auto; }

.bp3-button.bp3-align-left .bp3-button-text, .bp3-button.bp3-align-right .bp3-button-text,
.bp3-button-group.bp3-align-left .bp3-button-text,
.bp3-button-group.bp3-align-right .bp3-button-text{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto; }
.bp3-button-group{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex; }
  .bp3-button-group .bp3-button{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    position:relative;
    z-index:4; }
    .bp3-button-group .bp3-button:focus{
      z-index:5; }
    .bp3-button-group .bp3-button:hover{
      z-index:6; }
    .bp3-button-group .bp3-button:active, .bp3-button-group .bp3-button.bp3-active{
      z-index:7; }
    .bp3-button-group .bp3-button:disabled, .bp3-button-group .bp3-button.bp3-disabled{
      z-index:3; }
    .bp3-button-group .bp3-button[class*="bp3-intent-"]{
      z-index:9; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:focus{
        z-index:10; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:hover{
        z-index:11; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:active, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-active{
        z-index:12; }
      .bp3-button-group .bp3-button[class*="bp3-intent-"]:disabled, .bp3-button-group .bp3-button[class*="bp3-intent-"].bp3-disabled{
        z-index:8; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:first-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:first-child){
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    border-bottom-right-radius:0;
    border-top-right-radius:0;
    margin-right:-1px; }
  .bp3-button-group.bp3-minimal .bp3-button{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-button-group.bp3-minimal .bp3-button:hover{
      background:rgba(167, 182, 194, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026;
      text-decoration:none; }
    .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
      background:rgba(115, 134, 148, 0.3);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#182026; }
    .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
      background:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed; }
      .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
        background:rgba(115, 134, 148, 0.3); }
    .bp3-dark .bp3-button-group.bp3-minimal .bp3-button{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:inherit; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:hover{
        background:rgba(138, 155, 168, 0.15); }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-active{
        background:rgba(138, 155, 168, 0.3);
        color:#f5f8fa; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover{
        background:none;
        color:rgba(167, 182, 194, 0.6);
        cursor:not-allowed; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button:disabled:hover.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-disabled:hover.bp3-active{
          background:rgba(138, 155, 168, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
      color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.15);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#106ba3; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(16, 107, 163, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
        stroke:#106ba3; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary{
        color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:hover{
          background:rgba(19, 124, 189, 0.2);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-active{
          background:rgba(19, 124, 189, 0.3);
          color:#48aff0; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled{
          background:none;
          color:rgba(72, 175, 240, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-primary.bp3-disabled.bp3-active{
            background:rgba(19, 124, 189, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
      color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.15);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#0d8050; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(13, 128, 80, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
        stroke:#0d8050; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success{
        color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:hover{
          background:rgba(15, 153, 96, 0.2);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-active{
          background:rgba(15, 153, 96, 0.3);
          color:#3dcc91; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled{
          background:none;
          color:rgba(61, 204, 145, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-success.bp3-disabled.bp3-active{
            background:rgba(15, 153, 96, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
      color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.15);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#bf7326; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(191, 115, 38, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
        stroke:#bf7326; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning{
        color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:hover{
          background:rgba(217, 130, 43, 0.2);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-active{
          background:rgba(217, 130, 43, 0.3);
          color:#ffb366; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled{
          background:none;
          color:rgba(255, 179, 102, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-warning.bp3-disabled.bp3-active{
            background:rgba(217, 130, 43, 0.3); }
    .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
      color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.15);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#c23030; }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(194, 48, 48, 0.5); }
        .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }
      .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
        stroke:#c23030; }
      .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger{
        color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:hover{
          background:rgba(219, 55, 55, 0.2);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-active{
          background:rgba(219, 55, 55, 0.3);
          color:#ff7373; }
        .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled{
          background:none;
          color:rgba(255, 115, 115, 0.5); }
          .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-button-group.bp3-minimal .bp3-button.bp3-intent-danger.bp3-disabled.bp3-active{
            background:rgba(219, 55, 55, 0.3); }
  .bp3-button-group .bp3-popover-wrapper,
  .bp3-button-group .bp3-popover-target{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-button-group .bp3-button.bp3-fill,
  .bp3-button-group.bp3-fill .bp3-button:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-button-group.bp3-vertical{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column;
    vertical-align:top; }
    .bp3-button-group.bp3-vertical.bp3-fill{
      height:100%;
      width:unset; }
    .bp3-button-group.bp3-vertical .bp3-button{
      margin-right:0 !important;
      width:100%; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:first-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:first-child{
      border-radius:3px 3px 0 0; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:last-child .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:last-child{
      border-radius:0 0 3px 3px; }
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
    .bp3-button-group.bp3-vertical:not(.bp3-minimal) > .bp3-button:not(:last-child){
      margin-bottom:-1px; }
  .bp3-button-group.bp3-align-left .bp3-button{
    text-align:left; }
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group:not(.bp3-minimal) > .bp3-button:not(:last-child){
    margin-right:1px; }
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-popover-wrapper:not(:last-child) .bp3-button,
  .bp3-dark .bp3-button-group.bp3-vertical > .bp3-button:not(:last-child){
    margin-bottom:1px; }
.bp3-callout{
  font-size:14px;
  line-height:1.5;
  background-color:rgba(138, 155, 168, 0.15);
  border-radius:3px;
  padding:10px 12px 9px;
  position:relative;
  width:100%; }
  .bp3-callout[class*="bp3-icon-"]{
    padding-left:40px; }
    .bp3-callout[class*="bp3-icon-"]::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout.bp3-callout-icon{
    padding-left:40px; }
    .bp3-callout.bp3-callout-icon > .bp3-icon:first-child{
      color:#5c7080;
      left:10px;
      position:absolute;
      top:10px; }
  .bp3-callout .bp3-heading{
    line-height:20px;
    margin-bottom:5px;
    margin-top:0; }
    .bp3-callout .bp3-heading:last-child{
      margin-bottom:0; }
  .bp3-dark .bp3-callout{
    background-color:rgba(138, 155, 168, 0.2); }
    .bp3-dark .bp3-callout[class*="bp3-icon-"]::before{
      color:#a7b6c2; }
  .bp3-callout.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15); }
    .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-primary .bp3-heading{
      color:#106ba3; }
    .bp3-dark .bp3-callout.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-primary[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-primary > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-primary .bp3-heading{
        color:#48aff0; }
  .bp3-callout.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15); }
    .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-success .bp3-heading{
      color:#0d8050; }
    .bp3-dark .bp3-callout.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-success[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-success > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-success .bp3-heading{
        color:#3dcc91; }
  .bp3-callout.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15); }
    .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-warning .bp3-heading{
      color:#bf7326; }
    .bp3-dark .bp3-callout.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-warning[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-warning > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-warning .bp3-heading{
        color:#ffb366; }
  .bp3-callout.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15); }
    .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
    .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
    .bp3-callout.bp3-intent-danger .bp3-heading{
      color:#c23030; }
    .bp3-dark .bp3-callout.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25); }
      .bp3-dark .bp3-callout.bp3-intent-danger[class*="bp3-icon-"]::before,
      .bp3-dark .bp3-callout.bp3-intent-danger > .bp3-icon:first-child,
      .bp3-dark .bp3-callout.bp3-intent-danger .bp3-heading{
        color:#ff7373; }
  .bp3-running-text .bp3-callout{
    margin:20px 0; }
.bp3-card{
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
  padding:20px;
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-card.bp3-dark,
  .bp3-dark .bp3-card{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-0{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.15), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }
  .bp3-elevation-0.bp3-dark,
  .bp3-dark .bp3-elevation-0{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), 0 0 0 rgba(16, 22, 26, 0), 0 0 0 rgba(16, 22, 26, 0); }

.bp3-elevation-1{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-1.bp3-dark,
  .bp3-dark .bp3-elevation-1{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-elevation-2{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 1px 1px rgba(16, 22, 26, 0.2), 0 2px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-2.bp3-dark,
  .bp3-dark .bp3-elevation-2{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.4), 0 2px 6px rgba(16, 22, 26, 0.4); }

.bp3-elevation-3{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-3.bp3-dark,
  .bp3-dark .bp3-elevation-3{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-elevation-4{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2); }
  .bp3-elevation-4.bp3-dark,
  .bp3-dark .bp3-elevation-4{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:hover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  cursor:pointer; }
  .bp3-card.bp3-interactive:hover.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:hover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }

.bp3-card.bp3-interactive:active{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  opacity:0.9;
  -webkit-transition-duration:0;
          transition-duration:0; }
  .bp3-card.bp3-interactive:active.bp3-dark,
  .bp3-dark .bp3-card.bp3-interactive:active{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-collapse{
  height:0;
  overflow-y:hidden;
  -webkit-transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:height 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-collapse .bp3-collapse-body{
    -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-collapse .bp3-collapse-body[aria-hidden="true"]{
      display:none; }

.bp3-context-menu .bp3-popover-target{
  display:block; }

.bp3-context-menu-popover-target{
  position:fixed; }

.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-dialog-container{
  opacity:1;
  -webkit-transform:scale(1);
          transform:scale(1);
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  min-height:100%;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  width:100%; }
  .bp3-dialog-container.bp3-overlay-enter > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5); }
  .bp3-dialog-container.bp3-overlay-enter-active > .bp3-dialog, .bp3-dialog-container.bp3-overlay-appear-active > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-dialog-container.bp3-overlay-exit > .bp3-dialog{
    opacity:1;
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-dialog-container.bp3-overlay-exit-active > .bp3-dialog{
    opacity:0;
    -webkit-transform:scale(0.5);
            transform:scale(0.5);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-transform;
    transition-property:opacity, -webkit-transform;
    transition-property:opacity, transform;
    transition-property:opacity, transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }

.bp3-dialog{
  background:#ebf1f5;
  border-radius:6px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:30px 0;
  padding-bottom:20px;
  pointer-events:all;
  -webkit-user-select:text;
     -moz-user-select:text;
      -ms-user-select:text;
          user-select:text;
  width:500px; }
  .bp3-dialog:focus{
    outline:0; }
  .bp3-dialog.bp3-dark,
  .bp3-dark .bp3-dialog{
    background:#293742;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-dialog-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background:#ffffff;
  border-radius:6px 6px 0 0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding-left:20px;
  padding-right:5px;
  z-index:30; }
  .bp3-dialog-header .bp3-icon-large,
  .bp3-dialog-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-dialog-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-dialog-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-dialog-header{
    background:#30404d;
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-dialog-header .bp3-icon-large,
    .bp3-dark .bp3-dialog-header .bp3-icon{
      color:#a7b6c2; }

.bp3-dialog-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  margin:20px; }

.bp3-dialog-footer{
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  margin:0 20px; }

.bp3-dialog-footer-actions{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:end;
      -ms-flex-pack:end;
          justify-content:flex-end; }
  .bp3-dialog-footer-actions .bp3-button{
    margin-left:10px; }
.bp3-multistep-dialog-panels{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }

.bp3-multistep-dialog-left-panel{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column; }
  .bp3-dark .bp3-multistep-dialog-left-panel{
    background:#202b33; }

.bp3-multistep-dialog-right-panel{
  background-color:#f5f8fa;
  border-left:1px solid rgba(16, 22, 26, 0.15);
  border-radius:0 0 6px 0;
  -webkit-box-flex:3;
      -ms-flex:3;
          flex:3;
  min-width:0; }
  .bp3-dark .bp3-multistep-dialog-right-panel{
    background-color:#293742;
    border-left:1px solid rgba(16, 22, 26, 0.4); }

.bp3-multistep-dialog-footer{
  background-color:#ffffff;
  border-radius:0 0 6px 0;
  border-top:1px solid rgba(16, 22, 26, 0.15);
  padding:10px; }
  .bp3-dark .bp3-multistep-dialog-footer{
    background:#30404d;
    border-top:1px solid rgba(16, 22, 26, 0.4); }

.bp3-dialog-step-container{
  background-color:#f5f8fa;
  border-bottom:1px solid rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-dialog-step-container{
    background:#293742;
    border-bottom:1px solid rgba(16, 22, 26, 0.4); }
  .bp3-dialog-step-container.bp3-dialog-step-viewed{
    background-color:#ffffff; }
    .bp3-dark .bp3-dialog-step-container.bp3-dialog-step-viewed{
      background:#30404d; }

.bp3-dialog-step{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#f5f8fa;
  border-radius:6px;
  cursor:not-allowed;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:4px;
  padding:6px 14px; }
  .bp3-dark .bp3-dialog-step{
    background:#293742; }
  .bp3-dialog-step-viewed .bp3-dialog-step{
    background-color:#ffffff;
    cursor:pointer; }
    .bp3-dark .bp3-dialog-step-viewed .bp3-dialog-step{
      background:#30404d; }
  .bp3-dialog-step:hover{
    background-color:#f5f8fa; }
    .bp3-dark .bp3-dialog-step:hover{
      background:#293742; }

.bp3-dialog-step-icon{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:rgba(92, 112, 128, 0.6);
  border-radius:50%;
  color:#ffffff;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:25px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  width:25px; }
  .bp3-dark .bp3-dialog-step-icon{
    background-color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#2b95d6; }
  .bp3-dialog-step-viewed .bp3-dialog-step-icon{
    background-color:#8a9ba8; }

.bp3-dialog-step-title{
  color:rgba(92, 112, 128, 0.6);
  -webkit-box-flex:1;
      -ms-flex:1;
          flex:1;
  padding-left:10px; }
  .bp3-dark .bp3-dialog-step-title{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-active.bp3-dialog-step-viewed .bp3-dialog-step-title{
    color:#2b95d6; }
  .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
    color:#182026; }
    .bp3-dark .bp3-dialog-step-viewed:not(.bp3-active) .bp3-dialog-step-title{
      color:#f5f8fa; }
.bp3-drawer{
  background:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0;
  padding:0; }
  .bp3-drawer:focus{
    outline:0; }
  .bp3-drawer.bp3-position-top{
    height:50%;
    left:0;
    right:0;
    top:0; }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter, .bp3-drawer.bp3-position-top.bp3-overlay-appear{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%); }
    .bp3-drawer.bp3-position-top.bp3-overlay-enter-active, .bp3-drawer.bp3-position-top.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-top.bp3-overlay-exit-active{
      -webkit-transform:translateY(-100%);
              transform:translateY(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-bottom{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-enter-active, .bp3-drawer.bp3-position-bottom.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer.bp3-position-bottom.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-left{
    bottom:0;
    left:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter, .bp3-drawer.bp3-position-left.bp3-overlay-appear{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%); }
    .bp3-drawer.bp3-position-left.bp3-overlay-enter-active, .bp3-drawer.bp3-position-left.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-left.bp3-overlay-exit-active{
      -webkit-transform:translateX(-100%);
              transform:translateX(-100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-position-right{
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter, .bp3-drawer.bp3-position-right.bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer.bp3-position-right.bp3-overlay-enter-active, .bp3-drawer.bp3-position-right.bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer.bp3-position-right.bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right):not(.bp3-vertical){
    bottom:0;
    right:0;
    top:0;
    width:50%; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear{
      -webkit-transform:translateX(100%);
              transform:translateX(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-appear-active{
      -webkit-transform:translateX(0);
              transform:translateX(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit{
      -webkit-transform:translateX(0);
              transform:translateX(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right):not(.bp3-vertical).bp3-overlay-exit-active{
      -webkit-transform:translateX(100%);
              transform:translateX(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
  .bp3-position-right).bp3-vertical{
    bottom:0;
    height:50%;
    left:0;
    right:0; }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear{
      -webkit-transform:translateY(100%);
              transform:translateY(100%); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-enter-active, .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-appear-active{
      -webkit-transform:translateY(0);
              transform:translateY(0);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:200ms;
              transition-duration:200ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit{
      -webkit-transform:translateY(0);
              transform:translateY(0); }
    .bp3-drawer:not(.bp3-position-top):not(.bp3-position-bottom):not(.bp3-position-left):not(
    .bp3-position-right).bp3-vertical.bp3-overlay-exit-active{
      -webkit-transform:translateY(100%);
              transform:translateY(100%);
      -webkit-transition-delay:0;
              transition-delay:0;
      -webkit-transition-duration:100ms;
              transition-duration:100ms;
      -webkit-transition-property:-webkit-transform;
      transition-property:-webkit-transform;
      transition-property:transform;
      transition-property:transform, -webkit-transform;
      -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
              transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-drawer.bp3-dark,
  .bp3-dark .bp3-drawer{
    background:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }

.bp3-drawer-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border-radius:0;
  -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:0 1px 0 rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  min-height:40px;
  padding:5px;
  padding-left:20px;
  position:relative; }
  .bp3-drawer-header .bp3-icon-large,
  .bp3-drawer-header .bp3-icon{
    color:#5c7080;
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    margin-right:10px; }
  .bp3-drawer-header .bp3-heading{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:inherit;
    margin:0; }
    .bp3-drawer-header .bp3-heading:last-child{
      margin-right:20px; }
  .bp3-dark .bp3-drawer-header{
    -webkit-box-shadow:0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:0 1px 0 rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-drawer-header .bp3-icon-large,
    .bp3-dark .bp3-drawer-header .bp3-icon{
      color:#a7b6c2; }

.bp3-drawer-body{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  line-height:18px;
  overflow:auto; }

.bp3-drawer-footer{
  -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  padding:10px 20px;
  position:relative; }
  .bp3-dark .bp3-drawer-footer{
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.4); }
.bp3-editable-text{
  cursor:text;
  display:inline-block;
  max-width:100%;
  position:relative;
  vertical-align:top;
  white-space:nowrap; }
  .bp3-editable-text::before{
    bottom:-3px;
    left:-3px;
    position:absolute;
    right:-3px;
    top:-3px;
    border-radius:3px;
    content:"";
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9), box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#137cbd; }
  .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(19, 124, 189, 0.4); }
  .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#0f9960; }
  .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px rgba(15, 153, 96, 0.4); }
  .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#d9822b; }
  .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px rgba(217, 130, 43, 0.4); }
  .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-input,
  .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#db3737; }
  .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px rgba(219, 55, 55, 0.4); }
  .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-editable-text:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(255, 255, 255, 0.15); }
  .bp3-dark .bp3-editable-text.bp3-editable-text-editing::before{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-disabled::before{
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary .bp3-editable-text-content{
    color:#48aff0; }
  .bp3-dark .bp3-editable-text.bp3-intent-primary:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4);
            box-shadow:0 0 0 0 rgba(72, 175, 240, 0), 0 0 0 0 rgba(72, 175, 240, 0), inset 0 0 0 1px rgba(72, 175, 240, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-primary.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #48aff0, 0 0 0 3px rgba(72, 175, 240, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success .bp3-editable-text-content{
    color:#3dcc91; }
  .bp3-dark .bp3-editable-text.bp3-intent-success:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4);
            box-shadow:0 0 0 0 rgba(61, 204, 145, 0), 0 0 0 0 rgba(61, 204, 145, 0), inset 0 0 0 1px rgba(61, 204, 145, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-success.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #3dcc91, 0 0 0 3px rgba(61, 204, 145, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning .bp3-editable-text-content{
    color:#ffb366; }
  .bp3-dark .bp3-editable-text.bp3-intent-warning:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4);
            box-shadow:0 0 0 0 rgba(255, 179, 102, 0), 0 0 0 0 rgba(255, 179, 102, 0), inset 0 0 0 1px rgba(255, 179, 102, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-warning.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ffb366, 0 0 0 3px rgba(255, 179, 102, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger .bp3-editable-text-content{
    color:#ff7373; }
  .bp3-dark .bp3-editable-text.bp3-intent-danger:hover::before{
    -webkit-box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4);
            box-shadow:0 0 0 0 rgba(255, 115, 115, 0), 0 0 0 0 rgba(255, 115, 115, 0), inset 0 0 0 1px rgba(255, 115, 115, 0.4); }
  .bp3-dark .bp3-editable-text.bp3-intent-danger.bp3-editable-text-editing::before{
    -webkit-box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #ff7373, 0 0 0 3px rgba(255, 115, 115, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-editable-text-input,
.bp3-editable-text-content{
  color:inherit;
  display:inherit;
  font:inherit;
  letter-spacing:inherit;
  max-width:inherit;
  min-width:inherit;
  position:relative;
  resize:none;
  text-transform:inherit;
  vertical-align:top; }

.bp3-editable-text-input{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0;
  white-space:pre-wrap;
  width:100%; }
  .bp3-editable-text-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-editable-text-input:focus{
    outline:none; }
  .bp3-editable-text-input::-ms-clear{
    display:none; }

.bp3-editable-text-content{
  overflow:hidden;
  padding-right:2px;
  text-overflow:ellipsis;
  white-space:pre; }
  .bp3-editable-text-editing > .bp3-editable-text-content{
    left:0;
    position:absolute;
    visibility:hidden; }
  .bp3-editable-text-placeholder > .bp3-editable-text-content{
    color:rgba(92, 112, 128, 0.6); }
    .bp3-dark .bp3-editable-text-placeholder > .bp3-editable-text-content{
      color:rgba(167, 182, 194, 0.6); }

.bp3-editable-text.bp3-multiline{
  display:block; }
  .bp3-editable-text.bp3-multiline .bp3-editable-text-content{
    overflow:auto;
    white-space:pre-wrap;
    word-wrap:break-word; }
.bp3-divider{
  border-bottom:1px solid rgba(16, 22, 26, 0.15);
  border-right:1px solid rgba(16, 22, 26, 0.15);
  margin:5px; }
  .bp3-dark .bp3-divider{
    border-color:rgba(16, 22, 26, 0.4); }
.bp3-control-group{
  -webkit-transform:translateZ(0);
          transform:translateZ(0);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:stretch;
      -ms-flex-align:stretch;
          align-items:stretch; }
  .bp3-control-group > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select,
  .bp3-control-group .bp3-input,
  .bp3-control-group .bp3-select{
    position:relative; }
  .bp3-control-group .bp3-input{
    border-radius:inherit;
    z-index:2; }
    .bp3-control-group .bp3-input:focus{
      border-radius:3px;
      z-index:14; }
    .bp3-control-group .bp3-input[class*="bp3-intent"]{
      z-index:13; }
      .bp3-control-group .bp3-input[class*="bp3-intent"]:focus{
        z-index:15; }
    .bp3-control-group .bp3-input[readonly], .bp3-control-group .bp3-input:disabled, .bp3-control-group .bp3-input.bp3-disabled{
      z-index:1; }
  .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input{
    z-index:13; }
    .bp3-control-group .bp3-input-group[class*="bp3-intent"] .bp3-input:focus{
      z-index:15; }
  .bp3-control-group .bp3-button,
  .bp3-control-group .bp3-html-select select,
  .bp3-control-group .bp3-select select{
    -webkit-transform:translateZ(0);
            transform:translateZ(0);
    border-radius:inherit;
    z-index:4; }
    .bp3-control-group .bp3-button:focus,
    .bp3-control-group .bp3-html-select select:focus,
    .bp3-control-group .bp3-select select:focus{
      z-index:5; }
    .bp3-control-group .bp3-button:hover,
    .bp3-control-group .bp3-html-select select:hover,
    .bp3-control-group .bp3-select select:hover{
      z-index:6; }
    .bp3-control-group .bp3-button:active,
    .bp3-control-group .bp3-html-select select:active,
    .bp3-control-group .bp3-select select:active{
      z-index:7; }
    .bp3-control-group .bp3-button[readonly], .bp3-control-group .bp3-button:disabled, .bp3-control-group .bp3-button.bp3-disabled,
    .bp3-control-group .bp3-html-select select[readonly],
    .bp3-control-group .bp3-html-select select:disabled,
    .bp3-control-group .bp3-html-select select.bp3-disabled,
    .bp3-control-group .bp3-select select[readonly],
    .bp3-control-group .bp3-select select:disabled,
    .bp3-control-group .bp3-select select.bp3-disabled{
      z-index:3; }
    .bp3-control-group .bp3-button[class*="bp3-intent"],
    .bp3-control-group .bp3-html-select select[class*="bp3-intent"],
    .bp3-control-group .bp3-select select[class*="bp3-intent"]{
      z-index:9; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:focus,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:focus{
        z-index:10; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:hover,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:hover{
        z-index:11; }
      .bp3-control-group .bp3-button[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:active,
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:active{
        z-index:12; }
      .bp3-control-group .bp3-button[class*="bp3-intent"][readonly], .bp3-control-group .bp3-button[class*="bp3-intent"]:disabled, .bp3-control-group .bp3-button[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-html-select select[class*="bp3-intent"].bp3-disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"][readonly],
      .bp3-control-group .bp3-select select[class*="bp3-intent"]:disabled,
      .bp3-control-group .bp3-select select[class*="bp3-intent"].bp3-disabled{
        z-index:8; }
  .bp3-control-group .bp3-input-group > .bp3-icon,
  .bp3-control-group .bp3-input-group > .bp3-button,
  .bp3-control-group .bp3-input-group > .bp3-input-left-container,
  .bp3-control-group .bp3-input-group > .bp3-input-action{
    z-index:16; }
  .bp3-control-group .bp3-select::after,
  .bp3-control-group .bp3-html-select::after,
  .bp3-control-group .bp3-select > .bp3-icon,
  .bp3-control-group .bp3-html-select > .bp3-icon{
    z-index:17; }
  .bp3-control-group .bp3-select:focus-within{
    z-index:5; }
  .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:-1px; }
  .bp3-control-group:not(.bp3-vertical) > .bp3-divider:not(:first-child){
    margin-left:6px; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > *:not(.bp3-divider){
    margin-right:0; }
  .bp3-dark .bp3-control-group:not(.bp3-vertical) > .bp3-button + .bp3-button{
    margin-left:1px; }
  .bp3-control-group .bp3-popover-wrapper,
  .bp3-control-group .bp3-popover-target{
    border-radius:inherit; }
  .bp3-control-group > :first-child{
    border-radius:3px 0 0 3px; }
  .bp3-control-group > :last-child{
    border-radius:0 3px 3px 0;
    margin-right:0; }
  .bp3-control-group > :only-child{
    border-radius:3px;
    margin-right:0; }
  .bp3-control-group .bp3-input-group .bp3-button{
    border-radius:3px; }
  .bp3-control-group .bp3-numeric-input:not(:first-child) .bp3-input-group{
    border-bottom-left-radius:0;
    border-top-left-radius:0; }
  .bp3-control-group.bp3-fill{
    width:100%; }
  .bp3-control-group > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-fill > *:not(.bp3-fixed){
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto; }
  .bp3-control-group.bp3-vertical{
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-control-group.bp3-vertical > *{
      margin-top:-1px; }
    .bp3-control-group.bp3-vertical > :first-child{
      border-radius:3px 3px 0 0;
      margin-top:0; }
    .bp3-control-group.bp3-vertical > :last-child{
      border-radius:0 0 3px 3px; }
.bp3-control{
  cursor:pointer;
  display:block;
  margin-bottom:10px;
  position:relative;
  text-transform:none; }
  .bp3-control input:checked ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control input:checked ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control:hover input:checked ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control input:not(:disabled):active:checked ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control input:disabled:checked ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control:not(.bp3-align-right){
    padding-left:26px; }
    .bp3-control:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-26px; }
  .bp3-control.bp3-align-right{
    padding-right:26px; }
    .bp3-control.bp3-align-right .bp3-control-indicator{
      margin-right:-26px; }
  .bp3-control.bp3-disabled{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-control.bp3-inline{
    display:inline-block;
    margin-right:20px; }
  .bp3-control input{
    left:0;
    opacity:0;
    position:absolute;
    top:0;
    z-index:-1; }
  .bp3-control .bp3-control-indicator{
    background-clip:padding-box;
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    border:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    cursor:pointer;
    display:inline-block;
    font-size:16px;
    height:1em;
    margin-right:10px;
    margin-top:-3px;
    position:relative;
    -webkit-user-select:none;
       -moz-user-select:none;
        -ms-user-select:none;
            user-select:none;
    vertical-align:middle;
    width:1em; }
    .bp3-control .bp3-control-indicator::before{
      content:"";
      display:block;
      height:1em;
      width:1em; }
  .bp3-control:hover .bp3-control-indicator{
    background-color:#ebf1f5; }
  .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
    background:#d8e1e8;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    cursor:not-allowed; }
  .bp3-control input:focus ~ .bp3-control-indicator{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:2px;
    -moz-outline-radius:6px; }
  .bp3-control.bp3-align-right .bp3-control-indicator{
    float:right;
    margin-left:10px;
    margin-top:1px; }
  .bp3-control.bp3-large{
    font-size:16px; }
    .bp3-control.bp3-large:not(.bp3-align-right){
      padding-left:30px; }
      .bp3-control.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
        margin-left:-30px; }
    .bp3-control.bp3-large.bp3-align-right{
      padding-right:30px; }
      .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
        margin-right:-30px; }
    .bp3-control.bp3-large .bp3-control-indicator{
      font-size:20px; }
    .bp3-control.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-top:0; }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    background-color:#137cbd;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.1)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
    color:#ffffff; }
  .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 -1px 0 rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background:#0e5a8a;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-dark .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox:hover input:indeterminate ~ .bp3-control-indicator{
    background-color:#106ba3;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-checkbox input:not(:disabled):active:indeterminate ~ .bp3-control-indicator{
    background-color:#0e5a8a;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-control.bp3-checkbox .bp3-control-indicator{
    border-radius:3px; }
  .bp3-control.bp3-checkbox input:checked ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 00-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0012 5z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-checkbox input:indeterminate ~ .bp3-control-indicator::before{
    background-image:url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill-rule='evenodd' clip-rule='evenodd' d='M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z' fill='white'/%3e%3c/svg%3e"); }
  .bp3-control.bp3-radio .bp3-control-indicator{
    border-radius:50%; }
  .bp3-control.bp3-radio input:checked ~ .bp3-control-indicator::before{
    background-image:radial-gradient(#ffffff, #ffffff 28%, transparent 32%); }
  .bp3-control.bp3-radio input:checked:disabled ~ .bp3-control-indicator::before{
    opacity:0.5; }
  .bp3-control.bp3-radio input:focus ~ .bp3-control-indicator{
    -moz-outline-radius:16px; }
  .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(167, 182, 194, 0.5); }
  .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(115, 134, 148, 0.5); }
  .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(92, 112, 128, 0.5); }
  .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(206, 217, 224, 0.5); }
    .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(19, 124, 189, 0.5); }
    .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(255, 255, 255, 0.8); }
  .bp3-control.bp3-switch:not(.bp3-align-right){
    padding-left:38px; }
    .bp3-control.bp3-switch:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-38px; }
  .bp3-control.bp3-switch.bp3-align-right{
    padding-right:38px; }
    .bp3-control.bp3-switch.bp3-align-right .bp3-control-indicator{
      margin-right:-38px; }
  .bp3-control.bp3-switch .bp3-control-indicator{
    border:none;
    border-radius:1.75em;
    -webkit-box-shadow:none !important;
            box-shadow:none !important;
    min-width:1.75em;
    -webkit-transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:background-color 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:auto; }
    .bp3-control.bp3-switch .bp3-control-indicator::before{
      background:#ffffff;
      border-radius:50%;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
      height:calc(1em - 4px);
      left:0;
      margin:2px;
      position:absolute;
      -webkit-transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      transition:left 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
      width:calc(1em - 4px); }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    left:calc(100% - 1em); }
  .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right){
    padding-left:45px; }
    .bp3-control.bp3-switch.bp3-large:not(.bp3-align-right) .bp3-control-indicator{
      margin-left:-45px; }
  .bp3-control.bp3-switch.bp3-large.bp3-align-right{
    padding-right:45px; }
    .bp3-control.bp3-switch.bp3-large.bp3-align-right .bp3-control-indicator{
      margin-right:-45px; }
  .bp3-dark .bp3-control.bp3-switch input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-control.bp3-switch:hover input ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.7); }
  .bp3-dark .bp3-control.bp3-switch input:not(:disabled):active ~ .bp3-control-indicator{
    background:rgba(16, 22, 26, 0.9); }
  .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator{
    background:rgba(57, 75, 89, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator{
    background:#137cbd; }
  .bp3-dark .bp3-control.bp3-switch:hover input:checked ~ .bp3-control-indicator{
    background:#106ba3; }
  .bp3-dark .bp3-control.bp3-switch input:checked:not(:disabled):active ~ .bp3-control-indicator{
    background:#0e5a8a; }
  .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator{
    background:rgba(14, 90, 138, 0.5); }
    .bp3-dark .bp3-control.bp3-switch input:checked:disabled ~ .bp3-control-indicator::before{
      background:rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch .bp3-control-indicator::before{
    background:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator::before{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-control.bp3-switch .bp3-switch-inner-text{
    font-size:0.7em;
    text-align:center; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:first-child{
    line-height:0;
    margin-left:0.5em;
    margin-right:1.2em;
    visibility:hidden; }
  .bp3-control.bp3-switch .bp3-control-indicator-child:last-child{
    line-height:1em;
    margin-left:1.2em;
    margin-right:0.5em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:first-child{
    line-height:1em;
    visibility:visible; }
  .bp3-control.bp3-switch input:checked ~ .bp3-control-indicator .bp3-control-indicator-child:last-child{
    line-height:0;
    visibility:hidden; }
  .bp3-dark .bp3-control{
    color:#f5f8fa; }
    .bp3-dark .bp3-control.bp3-disabled{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-control .bp3-control-indicator{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-control:hover .bp3-control-indicator{
      background-color:#30404d; }
    .bp3-dark .bp3-control input:not(:disabled):active ~ .bp3-control-indicator{
      background:#202b33;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-control input:disabled ~ .bp3-control-indicator{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      cursor:not-allowed; }
    .bp3-dark .bp3-control.bp3-checkbox input:disabled:checked ~ .bp3-control-indicator, .bp3-dark .bp3-control.bp3-checkbox input:disabled:indeterminate ~ .bp3-control-indicator{
      color:rgba(167, 182, 194, 0.6); }
.bp3-file-input{
  cursor:pointer;
  display:inline-block;
  height:30px;
  position:relative; }
  .bp3-file-input input{
    margin:0;
    min-width:200px;
    opacity:0; }
    .bp3-file-input input:disabled + .bp3-file-upload-input,
    .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
      background:rgba(206, 217, 224, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      resize:none; }
      .bp3-file-input input:disabled + .bp3-file-upload-input::after,
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
        background-color:rgba(206, 217, 224, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(92, 112, 128, 0.6);
        cursor:not-allowed;
        outline:none; }
        .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active:hover,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active,
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active:hover{
          background:rgba(206, 217, 224, 0.7); }
      .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input, .bp3-dark
      .bp3-file-input input.bp3-disabled + .bp3-file-upload-input{
        background:rgba(57, 75, 89, 0.5);
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after, .bp3-dark
        .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after{
          background-color:rgba(57, 75, 89, 0.5);
          background-image:none;
          -webkit-box-shadow:none;
                  box-shadow:none;
          color:rgba(167, 182, 194, 0.6); }
          .bp3-dark .bp3-file-input input:disabled + .bp3-file-upload-input::after.bp3-active, .bp3-dark
          .bp3-file-input input.bp3-disabled + .bp3-file-upload-input::after.bp3-active{
            background:rgba(57, 75, 89, 0.7); }
  .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#182026; }
  .bp3-dark .bp3-file-input.bp3-file-input-has-selection .bp3-file-upload-input{
    color:#f5f8fa; }
  .bp3-file-input.bp3-fill{
    width:100%; }
  .bp3-file-input.bp3-large,
  .bp3-large .bp3-file-input{
    height:40px; }
  .bp3-file-input .bp3-file-upload-input-custom-text::after{
    content:attr(bp3-button-text); }

.bp3-file-upload-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:rgba(92, 112, 128, 0.6);
  left:0;
  padding-right:80px;
  position:absolute;
  right:0;
  top:0;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-file-upload-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-file-upload-input:focus, .bp3-file-upload-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-file-upload-input[type="search"], .bp3-file-upload-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-file-upload-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-file-upload-input:disabled, .bp3-file-upload-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-file-upload-input::after{
    background-color:#f5f8fa;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    color:#182026;
    min-height:24px;
    min-width:24px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    border-radius:3px;
    content:"Browse";
    line-height:24px;
    margin:3px;
    position:absolute;
    right:0;
    text-align:center;
    top:0;
    width:70px; }
    .bp3-file-upload-input::after:hover{
      background-clip:padding-box;
      background-color:#ebf1f5;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
    .bp3-file-upload-input::after:active, .bp3-file-upload-input::after.bp3-active{
      background-color:#d8e1e8;
      background-image:none;
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-file-upload-input::after:disabled, .bp3-file-upload-input::after.bp3-disabled{
      background-color:rgba(206, 217, 224, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(92, 112, 128, 0.6);
      cursor:not-allowed;
      outline:none; }
      .bp3-file-upload-input::after:disabled.bp3-active, .bp3-file-upload-input::after:disabled.bp3-active:hover, .bp3-file-upload-input::after.bp3-disabled.bp3-active, .bp3-file-upload-input::after.bp3-disabled.bp3-active:hover{
        background:rgba(206, 217, 224, 0.7); }
  .bp3-file-upload-input:hover::after{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-file-upload-input:active::after{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-large .bp3-file-upload-input{
    font-size:16px;
    height:40px;
    line-height:40px;
    padding-right:95px; }
    .bp3-large .bp3-file-upload-input[type="search"], .bp3-large .bp3-file-upload-input.bp3-round{
      padding:0 15px; }
    .bp3-large .bp3-file-upload-input::after{
      min-height:30px;
      min-width:30px;
      line-height:30px;
      margin:5px;
      width:85px; }
  .bp3-dark .bp3-file-upload-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:disabled, .bp3-dark .bp3-file-upload-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-file-upload-input::after{
      background-color:#394b59;
      background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
      background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
      color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover, .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        color:#f5f8fa; }
      .bp3-dark .bp3-file-upload-input::after:hover{
        background-color:#30404d;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-file-upload-input::after:active, .bp3-dark .bp3-file-upload-input::after.bp3-active{
        background-color:#202b33;
        background-image:none;
        -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
                box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
      .bp3-dark .bp3-file-upload-input::after:disabled, .bp3-dark .bp3-file-upload-input::after.bp3-disabled{
        background-color:rgba(57, 75, 89, 0.5);
        background-image:none;
        -webkit-box-shadow:none;
                box-shadow:none;
        color:rgba(167, 182, 194, 0.6); }
        .bp3-dark .bp3-file-upload-input::after:disabled.bp3-active, .bp3-dark .bp3-file-upload-input::after.bp3-disabled.bp3-active{
          background:rgba(57, 75, 89, 0.7); }
      .bp3-dark .bp3-file-upload-input::after .bp3-button-spinner .bp3-spinner-head{
        background:rgba(16, 22, 26, 0.5);
        stroke:#8a9ba8; }
    .bp3-dark .bp3-file-upload-input:hover::after{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-file-upload-input:active::after{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
.bp3-file-upload-input::after{
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
.bp3-form-group{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin:0 0 15px; }
  .bp3-form-group label.bp3-label{
    margin-bottom:5px; }
  .bp3-form-group .bp3-control{
    margin-top:7px; }
  .bp3-form-group .bp3-form-helper-text{
    color:#5c7080;
    font-size:12px;
    margin-top:5px; }
  .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#106ba3; }
  .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#0d8050; }
  .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#bf7326; }
  .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#c23030; }
  .bp3-form-group.bp3-inline{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row; }
    .bp3-form-group.bp3-inline.bp3-large label.bp3-label{
      line-height:40px;
      margin:0 10px 0 0; }
    .bp3-form-group.bp3-inline label.bp3-label{
      line-height:30px;
      margin:0 10px 0 0; }
  .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-dark .bp3-form-group.bp3-intent-primary .bp3-form-helper-text{
    color:#48aff0; }
  .bp3-dark .bp3-form-group.bp3-intent-success .bp3-form-helper-text{
    color:#3dcc91; }
  .bp3-dark .bp3-form-group.bp3-intent-warning .bp3-form-helper-text{
    color:#ffb366; }
  .bp3-dark .bp3-form-group.bp3-intent-danger .bp3-form-helper-text{
    color:#ff7373; }
  .bp3-dark .bp3-form-group .bp3-form-helper-text{
    color:#a7b6c2; }
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-label,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-text-muted,
  .bp3-dark .bp3-form-group.bp3-disabled .bp3-form-helper-text{
    color:rgba(167, 182, 194, 0.6) !important; }
.bp3-input-group{
  display:block;
  position:relative; }
  .bp3-input-group .bp3-input{
    position:relative;
    width:100%; }
    .bp3-input-group .bp3-input:not(:first-child){
      padding-left:30px; }
    .bp3-input-group .bp3-input:not(:last-child){
      padding-right:30px; }
  .bp3-input-group .bp3-input-action,
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-button,
  .bp3-input-group > .bp3-icon{
    position:absolute;
    top:0; }
    .bp3-input-group .bp3-input-action:first-child,
    .bp3-input-group > .bp3-input-left-container:first-child,
    .bp3-input-group > .bp3-button:first-child,
    .bp3-input-group > .bp3-icon:first-child{
      left:0; }
    .bp3-input-group .bp3-input-action:last-child,
    .bp3-input-group > .bp3-input-left-container:last-child,
    .bp3-input-group > .bp3-button:last-child,
    .bp3-input-group > .bp3-icon:last-child{
      right:0; }
  .bp3-input-group .bp3-button{
    min-height:24px;
    min-width:24px;
    margin:3px;
    padding:0 7px; }
    .bp3-input-group .bp3-button:empty{
      padding:0; }
  .bp3-input-group > .bp3-input-left-container,
  .bp3-input-group > .bp3-icon{
    z-index:1; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon{
    color:#5c7080; }
    .bp3-input-group > .bp3-input-left-container > .bp3-icon:empty,
    .bp3-input-group > .bp3-icon:empty{
      font-family:"Icons16", sans-serif;
      font-size:16px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased; }
  .bp3-input-group > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group > .bp3-icon,
  .bp3-input-group .bp3-input-action > .bp3-spinner{
    margin:7px; }
  .bp3-input-group .bp3-tag{
    margin:5px; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus),
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
    color:#5c7080; }
    .bp3-dark .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus), .bp3-dark
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus){
      color:#a7b6c2; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:not(:hover):not(:focus) .bp3-icon-large{
      color:#5c7080; }
  .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled,
  .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled{
    color:rgba(92, 112, 128, 0.6) !important; }
    .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-standard, .bp3-input-group .bp3-input:not(:focus) + .bp3-button.bp3-minimal:disabled .bp3-icon-large,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-standard,
    .bp3-input-group .bp3-input:not(:focus) + .bp3-input-action .bp3-button.bp3-minimal:disabled .bp3-icon-large{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-input-group.bp3-disabled{
    cursor:not-allowed; }
    .bp3-input-group.bp3-disabled .bp3-icon{
      color:rgba(92, 112, 128, 0.6); }
  .bp3-input-group.bp3-large .bp3-button{
    min-height:30px;
    min-width:30px;
    margin:5px; }
  .bp3-input-group.bp3-large > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-large > .bp3-icon,
  .bp3-input-group.bp3-large .bp3-input-action > .bp3-spinner{
    margin:12px; }
  .bp3-input-group.bp3-large .bp3-input{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input-group.bp3-large .bp3-input[type="search"], .bp3-input-group.bp3-large .bp3-input.bp3-round{
      padding:0 15px; }
    .bp3-input-group.bp3-large .bp3-input:not(:first-child){
      padding-left:40px; }
    .bp3-input-group.bp3-large .bp3-input:not(:last-child){
      padding-right:40px; }
  .bp3-input-group.bp3-small .bp3-button{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small .bp3-tag{
    min-height:20px;
    min-width:20px;
    margin:2px; }
  .bp3-input-group.bp3-small > .bp3-input-left-container > .bp3-icon,
  .bp3-input-group.bp3-small > .bp3-icon,
  .bp3-input-group.bp3-small .bp3-input-action > .bp3-spinner{
    margin:4px; }
  .bp3-input-group.bp3-small .bp3-input{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input-group.bp3-small .bp3-input[type="search"], .bp3-input-group.bp3-small .bp3-input.bp3-round{
      padding:0 12px; }
    .bp3-input-group.bp3-small .bp3-input:not(:first-child){
      padding-left:24px; }
    .bp3-input-group.bp3-small .bp3-input:not(:last-child){
      padding-right:24px; }
  .bp3-input-group.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-input-group.bp3-round .bp3-button,
  .bp3-input-group.bp3-round .bp3-input,
  .bp3-input-group.bp3-round .bp3-tag{
    border-radius:30px; }
  .bp3-dark .bp3-input-group .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-input-group.bp3-disabled .bp3-icon{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-input-group.bp3-intent-primary .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-primary .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input-group.bp3-intent-primary .bp3-input:disabled, .bp3-input-group.bp3-intent-primary .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-primary > .bp3-icon{
    color:#106ba3; }
    .bp3-dark .bp3-input-group.bp3-intent-primary > .bp3-icon{
      color:#48aff0; }
  .bp3-input-group.bp3-intent-success .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-success .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input-group.bp3-intent-success .bp3-input:disabled, .bp3-input-group.bp3-intent-success .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-success > .bp3-icon{
    color:#0d8050; }
    .bp3-dark .bp3-input-group.bp3-intent-success > .bp3-icon{
      color:#3dcc91; }
  .bp3-input-group.bp3-intent-warning .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-warning .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input-group.bp3-intent-warning .bp3-input:disabled, .bp3-input-group.bp3-intent-warning .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-warning > .bp3-icon{
    color:#bf7326; }
    .bp3-dark .bp3-input-group.bp3-intent-warning > .bp3-icon{
      color:#ffb366; }
  .bp3-input-group.bp3-intent-danger .bp3-input{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input-group.bp3-intent-danger .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input-group.bp3-intent-danger .bp3-input:disabled, .bp3-input-group.bp3-intent-danger .bp3-input.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-input-group.bp3-intent-danger > .bp3-icon{
    color:#c23030; }
    .bp3-dark .bp3-input-group.bp3-intent-danger > .bp3-icon{
      color:#ff7373; }
.bp3-input{
  -webkit-appearance:none;
     -moz-appearance:none;
          appearance:none;
  background:#ffffff;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
  color:#182026;
  font-size:14px;
  font-weight:400;
  height:30px;
  line-height:30px;
  outline:none;
  padding:0 10px;
  -webkit-transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-box-shadow 100ms cubic-bezier(0.4, 1, 0.75, 0.9);
  vertical-align:middle; }
  .bp3-input::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input:focus, .bp3-input.bp3-active{
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-input[type="search"], .bp3-input.bp3-round{
    border-radius:30px;
    -webkit-box-sizing:border-box;
            box-sizing:border-box;
    padding-left:10px; }
  .bp3-input[readonly]{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.15); }
  .bp3-input:disabled, .bp3-input.bp3-disabled{
    background:rgba(206, 217, 224, 0.5);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    resize:none; }
  .bp3-input.bp3-large{
    font-size:16px;
    height:40px;
    line-height:40px; }
    .bp3-input.bp3-large[type="search"], .bp3-input.bp3-large.bp3-round{
      padding:0 15px; }
  .bp3-input.bp3-small{
    font-size:12px;
    height:24px;
    line-height:24px;
    padding-left:8px;
    padding-right:8px; }
    .bp3-input.bp3-small[type="search"], .bp3-input.bp3-small.bp3-round{
      padding:0 12px; }
  .bp3-input.bp3-fill{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    width:100%; }
  .bp3-dark .bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-input:disabled, .bp3-dark .bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-input.bp3-intent-primary{
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-primary[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #137cbd;
              box-shadow:inset 0 0 0 1px #137cbd; }
    .bp3-input.bp3-intent-primary:disabled, .bp3-input.bp3-intent-primary.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px #137cbd, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary:focus{
        -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-primary[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #137cbd;
                box-shadow:inset 0 0 0 1px #137cbd; }
      .bp3-dark .bp3-input.bp3-intent-primary:disabled, .bp3-dark .bp3-input.bp3-intent-primary.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-success{
    -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success:focus{
      -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-success[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #0f9960;
              box-shadow:inset 0 0 0 1px #0f9960; }
    .bp3-input.bp3-intent-success:disabled, .bp3-input.bp3-intent-success.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-success{
      -webkit-box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), 0 0 0 0 rgba(15, 153, 96, 0), inset 0 0 0 1px #0f9960, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success:focus{
        -webkit-box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #0f9960, 0 0 0 1px #0f9960, 0 0 0 3px rgba(15, 153, 96, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-success[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #0f9960;
                box-shadow:inset 0 0 0 1px #0f9960; }
      .bp3-dark .bp3-input.bp3-intent-success:disabled, .bp3-dark .bp3-input.bp3-intent-success.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-warning{
    -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning:focus{
      -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-warning[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #d9822b;
              box-shadow:inset 0 0 0 1px #d9822b; }
    .bp3-input.bp3-intent-warning:disabled, .bp3-input.bp3-intent-warning.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), 0 0 0 0 rgba(217, 130, 43, 0), inset 0 0 0 1px #d9822b, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning:focus{
        -webkit-box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #d9822b, 0 0 0 1px #d9822b, 0 0 0 3px rgba(217, 130, 43, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-warning[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #d9822b;
                box-shadow:inset 0 0 0 1px #d9822b; }
      .bp3-dark .bp3-input.bp3-intent-warning:disabled, .bp3-dark .bp3-input.bp3-intent-warning.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input.bp3-intent-danger{
    -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.15), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger:focus{
      -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-input.bp3-intent-danger[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px #db3737;
              box-shadow:inset 0 0 0 1px #db3737; }
    .bp3-input.bp3-intent-danger:disabled, .bp3-input.bp3-intent-danger.bp3-disabled{
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-input.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), 0 0 0 0 rgba(219, 55, 55, 0), inset 0 0 0 1px #db3737, inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger:focus{
        -webkit-box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
                box-shadow:0 0 0 1px #db3737, 0 0 0 1px #db3737, 0 0 0 3px rgba(219, 55, 55, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
      .bp3-dark .bp3-input.bp3-intent-danger[readonly]{
        -webkit-box-shadow:inset 0 0 0 1px #db3737;
                box-shadow:inset 0 0 0 1px #db3737; }
      .bp3-dark .bp3-input.bp3-intent-danger:disabled, .bp3-dark .bp3-input.bp3-intent-danger.bp3-disabled{
        -webkit-box-shadow:none;
                box-shadow:none; }
  .bp3-input::-ms-clear{
    display:none; }
textarea.bp3-input{
  max-width:100%;
  padding:10px; }
  textarea.bp3-input, textarea.bp3-input.bp3-large, textarea.bp3-input.bp3-small{
    height:auto;
    line-height:inherit; }
  textarea.bp3-input.bp3-small{
    padding:8px; }
  .bp3-dark textarea.bp3-input{
    background:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), 0 0 0 0 rgba(19, 124, 189, 0), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark textarea.bp3-input::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input::placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark textarea.bp3-input:focus{
      -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input[readonly]{
      -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark textarea.bp3-input:disabled, .bp3-dark textarea.bp3-input.bp3-disabled{
      background:rgba(57, 75, 89, 0.5);
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
label.bp3-label{
  display:block;
  margin-bottom:15px;
  margin-top:0; }
  label.bp3-label .bp3-html-select,
  label.bp3-label .bp3-input,
  label.bp3-label .bp3-select,
  label.bp3-label .bp3-slider,
  label.bp3-label .bp3-popover-wrapper{
    display:block;
    margin-top:5px;
    text-transform:none; }
  label.bp3-label .bp3-button-group{
    margin-top:5px; }
  label.bp3-label .bp3-select select,
  label.bp3-label .bp3-html-select select{
    font-weight:400;
    vertical-align:top;
    width:100%; }
  label.bp3-label.bp3-disabled,
  label.bp3-label.bp3-disabled .bp3-text-muted{
    color:rgba(92, 112, 128, 0.6); }
  label.bp3-label.bp3-inline{
    line-height:30px; }
    label.bp3-label.bp3-inline .bp3-html-select,
    label.bp3-label.bp3-inline .bp3-input,
    label.bp3-label.bp3-inline .bp3-input-group,
    label.bp3-label.bp3-inline .bp3-select,
    label.bp3-label.bp3-inline .bp3-popover-wrapper{
      display:inline-block;
      margin:0 0 0 5px;
      vertical-align:top; }
    label.bp3-label.bp3-inline .bp3-button-group{
      margin:0 0 0 5px; }
    label.bp3-label.bp3-inline .bp3-input-group .bp3-input{
      margin-left:0; }
    label.bp3-label.bp3-inline.bp3-large{
      line-height:40px; }
  label.bp3-label:not(.bp3-inline) .bp3-popover-target{
    display:block; }
  .bp3-dark label.bp3-label{
    color:#f5f8fa; }
    .bp3-dark label.bp3-label.bp3-disabled,
    .bp3-dark label.bp3-label.bp3-disabled .bp3-text-muted{
      color:rgba(167, 182, 194, 0.6); }
.bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button{
  -webkit-box-flex:1;
      -ms-flex:1 1 14px;
          flex:1 1 14px;
  min-height:0;
  padding:0;
  width:30px; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:first-child{
    border-radius:0 3px 0 0; }
  .bp3-numeric-input .bp3-button-group.bp3-vertical > .bp3-button:last-child{
    border-radius:0 0 3px 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:first-child{
  border-radius:3px 0 0 0; }

.bp3-numeric-input .bp3-button-group.bp3-vertical:first-child > .bp3-button:last-child{
  border-radius:0 0 0 3px; }

.bp3-numeric-input.bp3-large .bp3-button-group.bp3-vertical > .bp3-button{
  width:40px; }

form{
  display:block; }
.bp3-html-select select,
.bp3-select select{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  border:none;
  border-radius:3px;
  cursor:pointer;
  font-size:14px;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  padding:5px 10px;
  text-align:left;
  vertical-align:middle;
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  -moz-appearance:none;
  -webkit-appearance:none;
  border-radius:3px;
  height:30px;
  padding:0 25px 0 10px;
  width:100%; }
  .bp3-html-select select > *, .bp3-select select > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-html-select select > .bp3-fill, .bp3-select select > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-html-select select::before,
  .bp3-select select::before, .bp3-html-select select > *, .bp3-select select > *{
    margin-right:7px; }
  .bp3-html-select select:empty::before,
  .bp3-select select:empty::before,
  .bp3-html-select select > :last-child,
  .bp3-select select > :last-child{
    margin-right:0; }
  .bp3-html-select select:hover,
  .bp3-select select:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-html-select select:active,
  .bp3-select select:active, .bp3-html-select select.bp3-active,
  .bp3-select select.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-html-select select:disabled,
  .bp3-select select:disabled, .bp3-html-select select.bp3-disabled,
  .bp3-select select.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-html-select select:disabled.bp3-active,
    .bp3-select select:disabled.bp3-active, .bp3-html-select select:disabled.bp3-active:hover,
    .bp3-select select:disabled.bp3-active:hover, .bp3-html-select select.bp3-disabled.bp3-active,
    .bp3-select select.bp3-disabled.bp3-active, .bp3-html-select select.bp3-disabled.bp3-active:hover,
    .bp3-select select.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }

.bp3-html-select.bp3-minimal select,
.bp3-select.bp3-minimal select{
  background:none;
  -webkit-box-shadow:none;
          box-shadow:none; }
  .bp3-html-select.bp3-minimal select:hover,
  .bp3-select.bp3-minimal select:hover{
    background:rgba(167, 182, 194, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026;
    text-decoration:none; }
  .bp3-html-select.bp3-minimal select:active,
  .bp3-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal select.bp3-active,
  .bp3-select.bp3-minimal select.bp3-active{
    background:rgba(115, 134, 148, 0.3);
    -webkit-box-shadow:none;
            box-shadow:none;
    color:#182026; }
  .bp3-html-select.bp3-minimal select:disabled,
  .bp3-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal select:disabled:hover,
  .bp3-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal select.bp3-disabled,
  .bp3-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal select.bp3-disabled:hover,
  .bp3-select.bp3-minimal select.bp3-disabled:hover{
    background:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
    .bp3-html-select.bp3-minimal select:disabled.bp3-active,
    .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active,
    .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active{
      background:rgba(115, 134, 148, 0.3); }
  .bp3-dark .bp3-html-select.bp3-minimal select, .bp3-html-select.bp3-minimal .bp3-dark select,
  .bp3-dark .bp3-select.bp3-minimal select, .bp3-select.bp3-minimal .bp3-dark select{
    background:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:inherit; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover, .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none; }
    .bp3-dark .bp3-html-select.bp3-minimal select:hover, .bp3-html-select.bp3-minimal .bp3-dark select:hover,
    .bp3-dark .bp3-select.bp3-minimal select:hover, .bp3-select.bp3-minimal .bp3-dark select:hover{
      background:rgba(138, 155, 168, 0.15); }
    .bp3-dark .bp3-html-select.bp3-minimal select:active, .bp3-html-select.bp3-minimal .bp3-dark select:active,
    .bp3-dark .bp3-select.bp3-minimal select:active, .bp3-select.bp3-minimal .bp3-dark select:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-active,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-active{
      background:rgba(138, 155, 168, 0.3);
      color:#f5f8fa; }
    .bp3-dark .bp3-html-select.bp3-minimal select:disabled, .bp3-html-select.bp3-minimal .bp3-dark select:disabled,
    .bp3-dark .bp3-select.bp3-minimal select:disabled, .bp3-select.bp3-minimal .bp3-dark select:disabled, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select:disabled:hover, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover{
      background:none;
      color:rgba(167, 182, 194, 0.6);
      cursor:not-allowed; }
      .bp3-dark .bp3-html-select.bp3-minimal select:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select:disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select:disabled:hover.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-disabled:hover.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-disabled:hover.bp3-active{
        background:rgba(138, 155, 168, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-primary,
  .bp3-select.bp3-minimal select.bp3-intent-primary{
    color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover,
    .bp3-select.bp3-minimal select.bp3-intent-primary:hover{
      background:rgba(19, 124, 189, 0.15);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:active,
    .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active{
      background:rgba(19, 124, 189, 0.3);
      color:#106ba3; }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled{
      background:none;
      color:rgba(16, 107, 163, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active{
        background:rgba(19, 124, 189, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-primary .bp3-button-spinner .bp3-spinner-head{
      stroke:#106ba3; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary{
      color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:hover{
        background:rgba(19, 124, 189, 0.2);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-active{
        background:rgba(19, 124, 189, 0.3);
        color:#48aff0; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled{
        background:none;
        color:rgba(72, 175, 240, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-primary.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-primary.bp3-disabled.bp3-active{
          background:rgba(19, 124, 189, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-success,
  .bp3-select.bp3-minimal select.bp3-intent-success{
    color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:hover,
    .bp3-select.bp3-minimal select.bp3-intent-success:hover{
      background:rgba(15, 153, 96, 0.15);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:active,
    .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active{
      background:rgba(15, 153, 96, 0.3);
      color:#0d8050; }
    .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled{
      background:none;
      color:rgba(13, 128, 80, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active{
        background:rgba(15, 153, 96, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-success .bp3-button-spinner .bp3-spinner-head{
      stroke:#0d8050; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success{
      color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:hover{
        background:rgba(15, 153, 96, 0.2);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-active{
        background:rgba(15, 153, 96, 0.3);
        color:#3dcc91; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled{
        background:none;
        color:rgba(61, 204, 145, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-success.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-success.bp3-disabled.bp3-active{
          background:rgba(15, 153, 96, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-warning,
  .bp3-select.bp3-minimal select.bp3-intent-warning{
    color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover,
    .bp3-select.bp3-minimal select.bp3-intent-warning:hover{
      background:rgba(217, 130, 43, 0.15);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:active,
    .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active{
      background:rgba(217, 130, 43, 0.3);
      color:#bf7326; }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled{
      background:none;
      color:rgba(191, 115, 38, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active{
        background:rgba(217, 130, 43, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-warning .bp3-button-spinner .bp3-spinner-head{
      stroke:#bf7326; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning{
      color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:hover{
        background:rgba(217, 130, 43, 0.2);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-active{
        background:rgba(217, 130, 43, 0.3);
        color:#ffb366; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled{
        background:none;
        color:rgba(255, 179, 102, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-warning.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-warning.bp3-disabled.bp3-active{
          background:rgba(217, 130, 43, 0.3); }
  .bp3-html-select.bp3-minimal select.bp3-intent-danger,
  .bp3-select.bp3-minimal select.bp3-intent-danger{
    color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover,
    .bp3-select.bp3-minimal select.bp3-intent-danger:hover{
      background:rgba(219, 55, 55, 0.15);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:active,
    .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active{
      background:rgba(219, 55, 55, 0.3);
      color:#c23030; }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled,
    .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled{
      background:none;
      color:rgba(194, 48, 48, 0.5); }
      .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active,
      .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active{
        background:rgba(219, 55, 55, 0.3); }
    .bp3-html-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head, .bp3-select.bp3-minimal select.bp3-intent-danger .bp3-button-spinner .bp3-spinner-head{
      stroke:#c23030; }
    .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger,
    .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger{
      color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:hover, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:hover{
        background:rgba(219, 55, 55, 0.2);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-active{
        background:rgba(219, 55, 55, 0.3);
        color:#ff7373; }
      .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled,
      .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled{
        background:none;
        color:rgba(255, 115, 115, 0.5); }
        .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger:disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger:disabled.bp3-active, .bp3-dark .bp3-html-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-html-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active,
        .bp3-dark .bp3-select.bp3-minimal select.bp3-intent-danger.bp3-disabled.bp3-active, .bp3-select.bp3-minimal .bp3-dark select.bp3-intent-danger.bp3-disabled.bp3-active{
          background:rgba(219, 55, 55, 0.3); }

.bp3-html-select.bp3-large select,
.bp3-select.bp3-large select{
  font-size:16px;
  height:40px;
  padding-right:35px; }

.bp3-dark .bp3-html-select select, .bp3-dark .bp3-select select{
  background-color:#394b59;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
  color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover, .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select select:hover, .bp3-dark .bp3-select select:hover{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-html-select select:active, .bp3-dark .bp3-select select:active, .bp3-dark .bp3-html-select select.bp3-active, .bp3-dark .bp3-select select.bp3-active{
    background-color:#202b33;
    background-image:none;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-html-select select:disabled, .bp3-dark .bp3-select select:disabled, .bp3-dark .bp3-html-select select.bp3-disabled, .bp3-dark .bp3-select select.bp3-disabled{
    background-color:rgba(57, 75, 89, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-html-select select:disabled.bp3-active, .bp3-dark .bp3-select select:disabled.bp3-active, .bp3-dark .bp3-html-select select.bp3-disabled.bp3-active, .bp3-dark .bp3-select select.bp3-disabled.bp3-active{
      background:rgba(57, 75, 89, 0.7); }
  .bp3-dark .bp3-html-select select .bp3-button-spinner .bp3-spinner-head, .bp3-dark .bp3-select select .bp3-button-spinner .bp3-spinner-head{
    background:rgba(16, 22, 26, 0.5);
    stroke:#8a9ba8; }

.bp3-html-select select:disabled,
.bp3-select select:disabled{
  background-color:rgba(206, 217, 224, 0.5);
  -webkit-box-shadow:none;
          box-shadow:none;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-html-select .bp3-icon,
.bp3-select .bp3-icon, .bp3-select::after{
  color:#5c7080;
  pointer-events:none;
  position:absolute;
  right:7px;
  top:7px; }
  .bp3-html-select .bp3-disabled.bp3-icon,
  .bp3-select .bp3-disabled.bp3-icon, .bp3-disabled.bp3-select::after{
    color:rgba(92, 112, 128, 0.6); }
.bp3-html-select,
.bp3-select{
  display:inline-block;
  letter-spacing:normal;
  position:relative;
  vertical-align:middle; }
  .bp3-html-select select::-ms-expand,
  .bp3-select select::-ms-expand{
    display:none; }
  .bp3-html-select .bp3-icon,
  .bp3-select .bp3-icon{
    color:#5c7080; }
    .bp3-html-select .bp3-icon:hover,
    .bp3-select .bp3-icon:hover{
      color:#182026; }
    .bp3-dark .bp3-html-select .bp3-icon, .bp3-dark
    .bp3-select .bp3-icon{
      color:#a7b6c2; }
      .bp3-dark .bp3-html-select .bp3-icon:hover, .bp3-dark
      .bp3-select .bp3-icon:hover{
        color:#f5f8fa; }
  .bp3-html-select.bp3-large::after,
  .bp3-html-select.bp3-large .bp3-icon,
  .bp3-select.bp3-large::after,
  .bp3-select.bp3-large .bp3-icon{
    right:12px;
    top:12px; }
  .bp3-html-select.bp3-fill,
  .bp3-html-select.bp3-fill select,
  .bp3-select.bp3-fill,
  .bp3-select.bp3-fill select{
    width:100%; }
  .bp3-dark .bp3-html-select option, .bp3-dark
  .bp3-select option{
    background-color:#30404d;
    color:#f5f8fa; }
  .bp3-dark .bp3-html-select option:disabled, .bp3-dark
  .bp3-select option:disabled{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-html-select::after, .bp3-dark
  .bp3-select::after{
    color:#a7b6c2; }

.bp3-select::after{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  content:""; }
.bp3-running-text table, table.bp3-html-table{
  border-spacing:0;
  font-size:14px; }
  .bp3-running-text table th, table.bp3-html-table th,
  .bp3-running-text table td,
  table.bp3-html-table td{
    padding:11px;
    text-align:left;
    vertical-align:top; }
  .bp3-running-text table th, table.bp3-html-table th{
    color:#182026;
    font-weight:600; }
  
  .bp3-running-text table td,
  table.bp3-html-table td{
    color:#182026; }
  .bp3-running-text table tbody tr:first-child th, table.bp3-html-table tbody tr:first-child th,
  .bp3-running-text table tbody tr:first-child td,
  table.bp3-html-table tbody tr:first-child td,
  .bp3-running-text table tfoot tr:first-child th,
  table.bp3-html-table tfoot tr:first-child th,
  .bp3-running-text table tfoot tr:first-child td,
  table.bp3-html-table tfoot tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  .bp3-dark .bp3-running-text table th, .bp3-running-text .bp3-dark table th, .bp3-dark table.bp3-html-table th{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table td, .bp3-running-text .bp3-dark table td, .bp3-dark table.bp3-html-table td{
    color:#f5f8fa; }
  .bp3-dark .bp3-running-text table tbody tr:first-child th, .bp3-running-text .bp3-dark table tbody tr:first-child th, .bp3-dark table.bp3-html-table tbody tr:first-child th,
  .bp3-dark .bp3-running-text table tbody tr:first-child td,
  .bp3-running-text .bp3-dark table tbody tr:first-child td,
  .bp3-dark table.bp3-html-table tbody tr:first-child td,
  .bp3-dark .bp3-running-text table tfoot tr:first-child th,
  .bp3-running-text .bp3-dark table tfoot tr:first-child th,
  .bp3-dark table.bp3-html-table tfoot tr:first-child th,
  .bp3-dark .bp3-running-text table tfoot tr:first-child td,
  .bp3-running-text .bp3-dark table tfoot tr:first-child td,
  .bp3-dark table.bp3-html-table tfoot tr:first-child td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }

table.bp3-html-table.bp3-html-table-condensed th,
table.bp3-html-table.bp3-html-table-condensed td, table.bp3-html-table.bp3-small th,
table.bp3-html-table.bp3-small td{
  padding-bottom:6px;
  padding-top:6px; }

table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
  background:rgba(191, 204, 214, 0.15); }

table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
  -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered tbody tr td,
table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
  -webkit-box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15);
          box-shadow:inset 0 1px 0 0 rgba(16, 22, 26, 0.15); }
  table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
  table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
    -webkit-box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 1px 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
  -webkit-box-shadow:none;
          box-shadow:none; }
  table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(16, 22, 26, 0.15); }

table.bp3-html-table.bp3-interactive tbody tr:hover td{
  background-color:rgba(191, 204, 214, 0.3);
  cursor:pointer; }

table.bp3-html-table.bp3-interactive tbody tr:active td{
  background-color:rgba(191, 204, 214, 0.4); }

.bp3-dark table.bp3-html-table{ }
  .bp3-dark table.bp3-html-table.bp3-html-table-striped tbody tr:nth-child(odd) td{
    background:rgba(92, 112, 128, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered th:not(:first-child){
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td,
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td{
    -webkit-box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 0 1px 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tbody tr td:not(:first-child),
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered tfoot tr td:not(:first-child){
      -webkit-box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15);
              box-shadow:inset 1px 1px 0 0 rgba(255, 255, 255, 0.15); }
  .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td{
    -webkit-box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15);
            box-shadow:inset 1px 0 0 0 rgba(255, 255, 255, 0.15); }
    .bp3-dark table.bp3-html-table.bp3-html-table-bordered.bp3-html-table-striped tbody tr:not(:first-child) td:first-child{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:hover td{
    background-color:rgba(92, 112, 128, 0.3);
    cursor:pointer; }
  .bp3-dark table.bp3-html-table.bp3-interactive tbody tr:active td{
    background-color:rgba(92, 112, 128, 0.4); }

.bp3-key-combo{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center; }
  .bp3-key-combo > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-key-combo > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-key-combo::before,
  .bp3-key-combo > *{
    margin-right:5px; }
  .bp3-key-combo:empty::before,
  .bp3-key-combo > :last-child{
    margin-right:0; }

.bp3-hotkey-dialog{
  padding-bottom:0;
  top:40px; }
  .bp3-hotkey-dialog .bp3-dialog-body{
    margin:0;
    padding:0; }
  .bp3-hotkey-dialog .bp3-hotkey-label{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1; }

.bp3-hotkey-column{
  margin:auto;
  max-height:80vh;
  overflow-y:auto;
  padding:30px; }
  .bp3-hotkey-column .bp3-heading{
    margin-bottom:20px; }
    .bp3-hotkey-column .bp3-heading:not(:first-child){
      margin-top:40px; }

.bp3-hotkey{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:justify;
      -ms-flex-pack:justify;
          justify-content:space-between;
  margin-left:0;
  margin-right:0; }
  .bp3-hotkey:not(:last-child){
    margin-bottom:10px; }
.bp3-icon{
  display:inline-block;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  vertical-align:text-bottom; }
  .bp3-icon:not(:empty)::before{
    content:"" !important;
    content:unset !important; }
  .bp3-icon > svg{
    display:block; }
    .bp3-icon > svg:not([fill]){
      fill:currentColor; }

.bp3-icon.bp3-intent-primary, .bp3-icon-standard.bp3-intent-primary, .bp3-icon-large.bp3-intent-primary{
  color:#106ba3; }
  .bp3-dark .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-icon-large.bp3-intent-primary{
    color:#48aff0; }

.bp3-icon.bp3-intent-success, .bp3-icon-standard.bp3-intent-success, .bp3-icon-large.bp3-intent-success{
  color:#0d8050; }
  .bp3-dark .bp3-icon.bp3-intent-success, .bp3-dark .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-icon-large.bp3-intent-success{
    color:#3dcc91; }

.bp3-icon.bp3-intent-warning, .bp3-icon-standard.bp3-intent-warning, .bp3-icon-large.bp3-intent-warning{
  color:#bf7326; }
  .bp3-dark .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-icon-large.bp3-intent-warning{
    color:#ffb366; }

.bp3-icon.bp3-intent-danger, .bp3-icon-standard.bp3-intent-danger, .bp3-icon-large.bp3-intent-danger{
  color:#c23030; }
  .bp3-dark .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-icon-large.bp3-intent-danger{
    color:#ff7373; }

span.bp3-icon-standard{
  font-family:"Icons16", sans-serif;
  font-size:16px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon-large{
  font-family:"Icons20", sans-serif;
  font-size:20px;
  font-style:normal;
  font-weight:400;
  line-height:1;
  -moz-osx-font-smoothing:grayscale;
  -webkit-font-smoothing:antialiased;
  display:inline-block; }

span.bp3-icon:empty{
  font-family:"Icons20";
  font-size:inherit;
  font-style:normal;
  font-weight:400;
  line-height:1; }
  span.bp3-icon:empty::before{
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased; }

.bp3-icon-add::before{
  content:""; }

.bp3-icon-add-column-left::before{
  content:""; }

.bp3-icon-add-column-right::before{
  content:""; }

.bp3-icon-add-row-bottom::before{
  content:""; }

.bp3-icon-add-row-top::before{
  content:""; }

.bp3-icon-add-to-artifact::before{
  content:""; }

.bp3-icon-add-to-folder::before{
  content:""; }

.bp3-icon-airplane::before{
  content:""; }

.bp3-icon-align-center::before{
  content:""; }

.bp3-icon-align-justify::before{
  content:""; }

.bp3-icon-align-left::before{
  content:""; }

.bp3-icon-align-right::before{
  content:""; }

.bp3-icon-alignment-bottom::before{
  content:""; }

.bp3-icon-alignment-horizontal-center::before{
  content:""; }

.bp3-icon-alignment-left::before{
  content:""; }

.bp3-icon-alignment-right::before{
  content:""; }

.bp3-icon-alignment-top::before{
  content:""; }

.bp3-icon-alignment-vertical-center::before{
  content:""; }

.bp3-icon-annotation::before{
  content:""; }

.bp3-icon-application::before{
  content:""; }

.bp3-icon-applications::before{
  content:""; }

.bp3-icon-archive::before{
  content:""; }

.bp3-icon-arrow-bottom-left::before{
  content:"↙"; }

.bp3-icon-arrow-bottom-right::before{
  content:"↘"; }

.bp3-icon-arrow-down::before{
  content:"↓"; }

.bp3-icon-arrow-left::before{
  content:"←"; }

.bp3-icon-arrow-right::before{
  content:"→"; }

.bp3-icon-arrow-top-left::before{
  content:"↖"; }

.bp3-icon-arrow-top-right::before{
  content:"↗"; }

.bp3-icon-arrow-up::before{
  content:"↑"; }

.bp3-icon-arrows-horizontal::before{
  content:"↔"; }

.bp3-icon-arrows-vertical::before{
  content:"↕"; }

.bp3-icon-asterisk::before{
  content:"*"; }

.bp3-icon-automatic-updates::before{
  content:""; }

.bp3-icon-badge::before{
  content:""; }

.bp3-icon-ban-circle::before{
  content:""; }

.bp3-icon-bank-account::before{
  content:""; }

.bp3-icon-barcode::before{
  content:""; }

.bp3-icon-blank::before{
  content:""; }

.bp3-icon-blocked-person::before{
  content:""; }

.bp3-icon-bold::before{
  content:""; }

.bp3-icon-book::before{
  content:""; }

.bp3-icon-bookmark::before{
  content:""; }

.bp3-icon-box::before{
  content:""; }

.bp3-icon-briefcase::before{
  content:""; }

.bp3-icon-bring-data::before{
  content:""; }

.bp3-icon-build::before{
  content:""; }

.bp3-icon-calculator::before{
  content:""; }

.bp3-icon-calendar::before{
  content:""; }

.bp3-icon-camera::before{
  content:""; }

.bp3-icon-caret-down::before{
  content:"⌄"; }

.bp3-icon-caret-left::before{
  content:"〈"; }

.bp3-icon-caret-right::before{
  content:"〉"; }

.bp3-icon-caret-up::before{
  content:"⌃"; }

.bp3-icon-cell-tower::before{
  content:""; }

.bp3-icon-changes::before{
  content:""; }

.bp3-icon-chart::before{
  content:""; }

.bp3-icon-chat::before{
  content:""; }

.bp3-icon-chevron-backward::before{
  content:""; }

.bp3-icon-chevron-down::before{
  content:""; }

.bp3-icon-chevron-forward::before{
  content:""; }

.bp3-icon-chevron-left::before{
  content:""; }

.bp3-icon-chevron-right::before{
  content:""; }

.bp3-icon-chevron-up::before{
  content:""; }

.bp3-icon-circle::before{
  content:""; }

.bp3-icon-circle-arrow-down::before{
  content:""; }

.bp3-icon-circle-arrow-left::before{
  content:""; }

.bp3-icon-circle-arrow-right::before{
  content:""; }

.bp3-icon-circle-arrow-up::before{
  content:""; }

.bp3-icon-citation::before{
  content:""; }

.bp3-icon-clean::before{
  content:""; }

.bp3-icon-clipboard::before{
  content:""; }

.bp3-icon-cloud::before{
  content:"☁"; }

.bp3-icon-cloud-download::before{
  content:""; }

.bp3-icon-cloud-upload::before{
  content:""; }

.bp3-icon-code::before{
  content:""; }

.bp3-icon-code-block::before{
  content:""; }

.bp3-icon-cog::before{
  content:""; }

.bp3-icon-collapse-all::before{
  content:""; }

.bp3-icon-column-layout::before{
  content:""; }

.bp3-icon-comment::before{
  content:""; }

.bp3-icon-comparison::before{
  content:""; }

.bp3-icon-compass::before{
  content:""; }

.bp3-icon-compressed::before{
  content:""; }

.bp3-icon-confirm::before{
  content:""; }

.bp3-icon-console::before{
  content:""; }

.bp3-icon-contrast::before{
  content:""; }

.bp3-icon-control::before{
  content:""; }

.bp3-icon-credit-card::before{
  content:""; }

.bp3-icon-cross::before{
  content:"✗"; }

.bp3-icon-crown::before{
  content:""; }

.bp3-icon-cube::before{
  content:""; }

.bp3-icon-cube-add::before{
  content:""; }

.bp3-icon-cube-remove::before{
  content:""; }

.bp3-icon-curved-range-chart::before{
  content:""; }

.bp3-icon-cut::before{
  content:""; }

.bp3-icon-dashboard::before{
  content:""; }

.bp3-icon-data-lineage::before{
  content:""; }

.bp3-icon-database::before{
  content:""; }

.bp3-icon-delete::before{
  content:""; }

.bp3-icon-delta::before{
  content:"Δ"; }

.bp3-icon-derive-column::before{
  content:""; }

.bp3-icon-desktop::before{
  content:""; }

.bp3-icon-diagnosis::before{
  content:""; }

.bp3-icon-diagram-tree::before{
  content:""; }

.bp3-icon-direction-left::before{
  content:""; }

.bp3-icon-direction-right::before{
  content:""; }

.bp3-icon-disable::before{
  content:""; }

.bp3-icon-document::before{
  content:""; }

.bp3-icon-document-open::before{
  content:""; }

.bp3-icon-document-share::before{
  content:""; }

.bp3-icon-dollar::before{
  content:"$"; }

.bp3-icon-dot::before{
  content:"•"; }

.bp3-icon-double-caret-horizontal::before{
  content:""; }

.bp3-icon-double-caret-vertical::before{
  content:""; }

.bp3-icon-double-chevron-down::before{
  content:""; }

.bp3-icon-double-chevron-left::before{
  content:""; }

.bp3-icon-double-chevron-right::before{
  content:""; }

.bp3-icon-double-chevron-up::before{
  content:""; }

.bp3-icon-doughnut-chart::before{
  content:""; }

.bp3-icon-download::before{
  content:""; }

.bp3-icon-drag-handle-horizontal::before{
  content:""; }

.bp3-icon-drag-handle-vertical::before{
  content:""; }

.bp3-icon-draw::before{
  content:""; }

.bp3-icon-drive-time::before{
  content:""; }

.bp3-icon-duplicate::before{
  content:""; }

.bp3-icon-edit::before{
  content:"✎"; }

.bp3-icon-eject::before{
  content:"⏏"; }

.bp3-icon-endorsed::before{
  content:""; }

.bp3-icon-envelope::before{
  content:"✉"; }

.bp3-icon-equals::before{
  content:""; }

.bp3-icon-eraser::before{
  content:""; }

.bp3-icon-error::before{
  content:""; }

.bp3-icon-euro::before{
  content:"€"; }

.bp3-icon-exchange::before{
  content:""; }

.bp3-icon-exclude-row::before{
  content:""; }

.bp3-icon-expand-all::before{
  content:""; }

.bp3-icon-export::before{
  content:""; }

.bp3-icon-eye-off::before{
  content:""; }

.bp3-icon-eye-on::before{
  content:""; }

.bp3-icon-eye-open::before{
  content:""; }

.bp3-icon-fast-backward::before{
  content:""; }

.bp3-icon-fast-forward::before{
  content:""; }

.bp3-icon-feed::before{
  content:""; }

.bp3-icon-feed-subscribed::before{
  content:""; }

.bp3-icon-film::before{
  content:""; }

.bp3-icon-filter::before{
  content:""; }

.bp3-icon-filter-keep::before{
  content:""; }

.bp3-icon-filter-list::before{
  content:""; }

.bp3-icon-filter-open::before{
  content:""; }

.bp3-icon-filter-remove::before{
  content:""; }

.bp3-icon-flag::before{
  content:"⚑"; }

.bp3-icon-flame::before{
  content:""; }

.bp3-icon-flash::before{
  content:""; }

.bp3-icon-floppy-disk::before{
  content:""; }

.bp3-icon-flow-branch::before{
  content:""; }

.bp3-icon-flow-end::before{
  content:""; }

.bp3-icon-flow-linear::before{
  content:""; }

.bp3-icon-flow-review::before{
  content:""; }

.bp3-icon-flow-review-branch::before{
  content:""; }

.bp3-icon-flows::before{
  content:""; }

.bp3-icon-folder-close::before{
  content:""; }

.bp3-icon-folder-new::before{
  content:""; }

.bp3-icon-folder-open::before{
  content:""; }

.bp3-icon-folder-shared::before{
  content:""; }

.bp3-icon-folder-shared-open::before{
  content:""; }

.bp3-icon-follower::before{
  content:""; }

.bp3-icon-following::before{
  content:""; }

.bp3-icon-font::before{
  content:""; }

.bp3-icon-fork::before{
  content:""; }

.bp3-icon-form::before{
  content:""; }

.bp3-icon-full-circle::before{
  content:""; }

.bp3-icon-full-stacked-chart::before{
  content:""; }

.bp3-icon-fullscreen::before{
  content:""; }

.bp3-icon-function::before{
  content:""; }

.bp3-icon-gantt-chart::before{
  content:""; }

.bp3-icon-geolocation::before{
  content:""; }

.bp3-icon-geosearch::before{
  content:""; }

.bp3-icon-git-branch::before{
  content:""; }

.bp3-icon-git-commit::before{
  content:""; }

.bp3-icon-git-merge::before{
  content:""; }

.bp3-icon-git-new-branch::before{
  content:""; }

.bp3-icon-git-pull::before{
  content:""; }

.bp3-icon-git-push::before{
  content:""; }

.bp3-icon-git-repo::before{
  content:""; }

.bp3-icon-glass::before{
  content:""; }

.bp3-icon-globe::before{
  content:""; }

.bp3-icon-globe-network::before{
  content:""; }

.bp3-icon-graph::before{
  content:""; }

.bp3-icon-graph-remove::before{
  content:""; }

.bp3-icon-greater-than::before{
  content:""; }

.bp3-icon-greater-than-or-equal-to::before{
  content:""; }

.bp3-icon-grid::before{
  content:""; }

.bp3-icon-grid-view::before{
  content:""; }

.bp3-icon-group-objects::before{
  content:""; }

.bp3-icon-grouped-bar-chart::before{
  content:""; }

.bp3-icon-hand::before{
  content:""; }

.bp3-icon-hand-down::before{
  content:""; }

.bp3-icon-hand-left::before{
  content:""; }

.bp3-icon-hand-right::before{
  content:""; }

.bp3-icon-hand-up::before{
  content:""; }

.bp3-icon-header::before{
  content:""; }

.bp3-icon-header-one::before{
  content:""; }

.bp3-icon-header-two::before{
  content:""; }

.bp3-icon-headset::before{
  content:""; }

.bp3-icon-heart::before{
  content:"♥"; }

.bp3-icon-heart-broken::before{
  content:""; }

.bp3-icon-heat-grid::before{
  content:""; }

.bp3-icon-heatmap::before{
  content:""; }

.bp3-icon-help::before{
  content:"?"; }

.bp3-icon-helper-management::before{
  content:""; }

.bp3-icon-highlight::before{
  content:""; }

.bp3-icon-history::before{
  content:""; }

.bp3-icon-home::before{
  content:"⌂"; }

.bp3-icon-horizontal-bar-chart::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-asc::before{
  content:""; }

.bp3-icon-horizontal-bar-chart-desc::before{
  content:""; }

.bp3-icon-horizontal-distribution::before{
  content:""; }

.bp3-icon-id-number::before{
  content:""; }

.bp3-icon-image-rotate-left::before{
  content:""; }

.bp3-icon-image-rotate-right::before{
  content:""; }

.bp3-icon-import::before{
  content:""; }

.bp3-icon-inbox::before{
  content:""; }

.bp3-icon-inbox-filtered::before{
  content:""; }

.bp3-icon-inbox-geo::before{
  content:""; }

.bp3-icon-inbox-search::before{
  content:""; }

.bp3-icon-inbox-update::before{
  content:""; }

.bp3-icon-info-sign::before{
  content:"ℹ"; }

.bp3-icon-inheritance::before{
  content:""; }

.bp3-icon-inner-join::before{
  content:""; }

.bp3-icon-insert::before{
  content:""; }

.bp3-icon-intersection::before{
  content:""; }

.bp3-icon-ip-address::before{
  content:""; }

.bp3-icon-issue::before{
  content:""; }

.bp3-icon-issue-closed::before{
  content:""; }

.bp3-icon-issue-new::before{
  content:""; }

.bp3-icon-italic::before{
  content:""; }

.bp3-icon-join-table::before{
  content:""; }

.bp3-icon-key::before{
  content:""; }

.bp3-icon-key-backspace::before{
  content:""; }

.bp3-icon-key-command::before{
  content:""; }

.bp3-icon-key-control::before{
  content:""; }

.bp3-icon-key-delete::before{
  content:""; }

.bp3-icon-key-enter::before{
  content:""; }

.bp3-icon-key-escape::before{
  content:""; }

.bp3-icon-key-option::before{
  content:""; }

.bp3-icon-key-shift::before{
  content:""; }

.bp3-icon-key-tab::before{
  content:""; }

.bp3-icon-known-vehicle::before{
  content:""; }

.bp3-icon-lab-test::before{
  content:""; }

.bp3-icon-label::before{
  content:""; }

.bp3-icon-layer::before{
  content:""; }

.bp3-icon-layers::before{
  content:""; }

.bp3-icon-layout::before{
  content:""; }

.bp3-icon-layout-auto::before{
  content:""; }

.bp3-icon-layout-balloon::before{
  content:""; }

.bp3-icon-layout-circle::before{
  content:""; }

.bp3-icon-layout-grid::before{
  content:""; }

.bp3-icon-layout-group-by::before{
  content:""; }

.bp3-icon-layout-hierarchy::before{
  content:""; }

.bp3-icon-layout-linear::before{
  content:""; }

.bp3-icon-layout-skew-grid::before{
  content:""; }

.bp3-icon-layout-sorted-clusters::before{
  content:""; }

.bp3-icon-learning::before{
  content:""; }

.bp3-icon-left-join::before{
  content:""; }

.bp3-icon-less-than::before{
  content:""; }

.bp3-icon-less-than-or-equal-to::before{
  content:""; }

.bp3-icon-lifesaver::before{
  content:""; }

.bp3-icon-lightbulb::before{
  content:""; }

.bp3-icon-link::before{
  content:""; }

.bp3-icon-list::before{
  content:"☰"; }

.bp3-icon-list-columns::before{
  content:""; }

.bp3-icon-list-detail-view::before{
  content:""; }

.bp3-icon-locate::before{
  content:""; }

.bp3-icon-lock::before{
  content:""; }

.bp3-icon-log-in::before{
  content:""; }

.bp3-icon-log-out::before{
  content:""; }

.bp3-icon-manual::before{
  content:""; }

.bp3-icon-manually-entered-data::before{
  content:""; }

.bp3-icon-map::before{
  content:""; }

.bp3-icon-map-create::before{
  content:""; }

.bp3-icon-map-marker::before{
  content:""; }

.bp3-icon-maximize::before{
  content:""; }

.bp3-icon-media::before{
  content:""; }

.bp3-icon-menu::before{
  content:""; }

.bp3-icon-menu-closed::before{
  content:""; }

.bp3-icon-menu-open::before{
  content:""; }

.bp3-icon-merge-columns::before{
  content:""; }

.bp3-icon-merge-links::before{
  content:""; }

.bp3-icon-minimize::before{
  content:""; }

.bp3-icon-minus::before{
  content:"−"; }

.bp3-icon-mobile-phone::before{
  content:""; }

.bp3-icon-mobile-video::before{
  content:""; }

.bp3-icon-moon::before{
  content:""; }

.bp3-icon-more::before{
  content:""; }

.bp3-icon-mountain::before{
  content:""; }

.bp3-icon-move::before{
  content:""; }

.bp3-icon-mugshot::before{
  content:""; }

.bp3-icon-multi-select::before{
  content:""; }

.bp3-icon-music::before{
  content:""; }

.bp3-icon-new-drawing::before{
  content:""; }

.bp3-icon-new-grid-item::before{
  content:""; }

.bp3-icon-new-layer::before{
  content:""; }

.bp3-icon-new-layers::before{
  content:""; }

.bp3-icon-new-link::before{
  content:""; }

.bp3-icon-new-object::before{
  content:""; }

.bp3-icon-new-person::before{
  content:""; }

.bp3-icon-new-prescription::before{
  content:""; }

.bp3-icon-new-text-box::before{
  content:""; }

.bp3-icon-ninja::before{
  content:""; }

.bp3-icon-not-equal-to::before{
  content:""; }

.bp3-icon-notifications::before{
  content:""; }

.bp3-icon-notifications-updated::before{
  content:""; }

.bp3-icon-numbered-list::before{
  content:""; }

.bp3-icon-numerical::before{
  content:""; }

.bp3-icon-office::before{
  content:""; }

.bp3-icon-offline::before{
  content:""; }

.bp3-icon-oil-field::before{
  content:""; }

.bp3-icon-one-column::before{
  content:""; }

.bp3-icon-outdated::before{
  content:""; }

.bp3-icon-page-layout::before{
  content:""; }

.bp3-icon-panel-stats::before{
  content:""; }

.bp3-icon-panel-table::before{
  content:""; }

.bp3-icon-paperclip::before{
  content:""; }

.bp3-icon-paragraph::before{
  content:""; }

.bp3-icon-path::before{
  content:""; }

.bp3-icon-path-search::before{
  content:""; }

.bp3-icon-pause::before{
  content:""; }

.bp3-icon-people::before{
  content:""; }

.bp3-icon-percentage::before{
  content:""; }

.bp3-icon-person::before{
  content:""; }

.bp3-icon-phone::before{
  content:"☎"; }

.bp3-icon-pie-chart::before{
  content:""; }

.bp3-icon-pin::before{
  content:""; }

.bp3-icon-pivot::before{
  content:""; }

.bp3-icon-pivot-table::before{
  content:""; }

.bp3-icon-play::before{
  content:""; }

.bp3-icon-plus::before{
  content:"+"; }

.bp3-icon-polygon-filter::before{
  content:""; }

.bp3-icon-power::before{
  content:""; }

.bp3-icon-predictive-analysis::before{
  content:""; }

.bp3-icon-prescription::before{
  content:""; }

.bp3-icon-presentation::before{
  content:""; }

.bp3-icon-print::before{
  content:"⎙"; }

.bp3-icon-projects::before{
  content:""; }

.bp3-icon-properties::before{
  content:""; }

.bp3-icon-property::before{
  content:""; }

.bp3-icon-publish-function::before{
  content:""; }

.bp3-icon-pulse::before{
  content:""; }

.bp3-icon-random::before{
  content:""; }

.bp3-icon-record::before{
  content:""; }

.bp3-icon-redo::before{
  content:""; }

.bp3-icon-refresh::before{
  content:""; }

.bp3-icon-regression-chart::before{
  content:""; }

.bp3-icon-remove::before{
  content:""; }

.bp3-icon-remove-column::before{
  content:""; }

.bp3-icon-remove-column-left::before{
  content:""; }

.bp3-icon-remove-column-right::before{
  content:""; }

.bp3-icon-remove-row-bottom::before{
  content:""; }

.bp3-icon-remove-row-top::before{
  content:""; }

.bp3-icon-repeat::before{
  content:""; }

.bp3-icon-reset::before{
  content:""; }

.bp3-icon-resolve::before{
  content:""; }

.bp3-icon-rig::before{
  content:""; }

.bp3-icon-right-join::before{
  content:""; }

.bp3-icon-ring::before{
  content:""; }

.bp3-icon-rotate-document::before{
  content:""; }

.bp3-icon-rotate-page::before{
  content:""; }

.bp3-icon-satellite::before{
  content:""; }

.bp3-icon-saved::before{
  content:""; }

.bp3-icon-scatter-plot::before{
  content:""; }

.bp3-icon-search::before{
  content:""; }

.bp3-icon-search-around::before{
  content:""; }

.bp3-icon-search-template::before{
  content:""; }

.bp3-icon-search-text::before{
  content:""; }

.bp3-icon-segmented-control::before{
  content:""; }

.bp3-icon-select::before{
  content:""; }

.bp3-icon-selection::before{
  content:"⦿"; }

.bp3-icon-send-to::before{
  content:""; }

.bp3-icon-send-to-graph::before{
  content:""; }

.bp3-icon-send-to-map::before{
  content:""; }

.bp3-icon-series-add::before{
  content:""; }

.bp3-icon-series-configuration::before{
  content:""; }

.bp3-icon-series-derived::before{
  content:""; }

.bp3-icon-series-filtered::before{
  content:""; }

.bp3-icon-series-search::before{
  content:""; }

.bp3-icon-settings::before{
  content:""; }

.bp3-icon-share::before{
  content:""; }

.bp3-icon-shield::before{
  content:""; }

.bp3-icon-shop::before{
  content:""; }

.bp3-icon-shopping-cart::before{
  content:""; }

.bp3-icon-signal-search::before{
  content:""; }

.bp3-icon-sim-card::before{
  content:""; }

.bp3-icon-slash::before{
  content:""; }

.bp3-icon-small-cross::before{
  content:""; }

.bp3-icon-small-minus::before{
  content:""; }

.bp3-icon-small-plus::before{
  content:""; }

.bp3-icon-small-tick::before{
  content:""; }

.bp3-icon-snowflake::before{
  content:""; }

.bp3-icon-social-media::before{
  content:""; }

.bp3-icon-sort::before{
  content:""; }

.bp3-icon-sort-alphabetical::before{
  content:""; }

.bp3-icon-sort-alphabetical-desc::before{
  content:""; }

.bp3-icon-sort-asc::before{
  content:""; }

.bp3-icon-sort-desc::before{
  content:""; }

.bp3-icon-sort-numerical::before{
  content:""; }

.bp3-icon-sort-numerical-desc::before{
  content:""; }

.bp3-icon-split-columns::before{
  content:""; }

.bp3-icon-square::before{
  content:""; }

.bp3-icon-stacked-chart::before{
  content:""; }

.bp3-icon-star::before{
  content:"★"; }

.bp3-icon-star-empty::before{
  content:"☆"; }

.bp3-icon-step-backward::before{
  content:""; }

.bp3-icon-step-chart::before{
  content:""; }

.bp3-icon-step-forward::before{
  content:""; }

.bp3-icon-stop::before{
  content:""; }

.bp3-icon-stopwatch::before{
  content:""; }

.bp3-icon-strikethrough::before{
  content:""; }

.bp3-icon-style::before{
  content:""; }

.bp3-icon-swap-horizontal::before{
  content:""; }

.bp3-icon-swap-vertical::before{
  content:""; }

.bp3-icon-symbol-circle::before{
  content:""; }

.bp3-icon-symbol-cross::before{
  content:""; }

.bp3-icon-symbol-diamond::before{
  content:""; }

.bp3-icon-symbol-square::before{
  content:""; }

.bp3-icon-symbol-triangle-down::before{
  content:""; }

.bp3-icon-symbol-triangle-up::before{
  content:""; }

.bp3-icon-tag::before{
  content:""; }

.bp3-icon-take-action::before{
  content:""; }

.bp3-icon-taxi::before{
  content:""; }

.bp3-icon-text-highlight::before{
  content:""; }

.bp3-icon-th::before{
  content:""; }

.bp3-icon-th-derived::before{
  content:""; }

.bp3-icon-th-disconnect::before{
  content:""; }

.bp3-icon-th-filtered::before{
  content:""; }

.bp3-icon-th-list::before{
  content:""; }

.bp3-icon-thumbs-down::before{
  content:""; }

.bp3-icon-thumbs-up::before{
  content:""; }

.bp3-icon-tick::before{
  content:"✓"; }

.bp3-icon-tick-circle::before{
  content:""; }

.bp3-icon-time::before{
  content:"⏲"; }

.bp3-icon-timeline-area-chart::before{
  content:""; }

.bp3-icon-timeline-bar-chart::before{
  content:""; }

.bp3-icon-timeline-events::before{
  content:""; }

.bp3-icon-timeline-line-chart::before{
  content:""; }

.bp3-icon-tint::before{
  content:""; }

.bp3-icon-torch::before{
  content:""; }

.bp3-icon-tractor::before{
  content:""; }

.bp3-icon-train::before{
  content:""; }

.bp3-icon-translate::before{
  content:""; }

.bp3-icon-trash::before{
  content:""; }

.bp3-icon-tree::before{
  content:""; }

.bp3-icon-trending-down::before{
  content:""; }

.bp3-icon-trending-up::before{
  content:""; }

.bp3-icon-truck::before{
  content:""; }

.bp3-icon-two-columns::before{
  content:""; }

.bp3-icon-unarchive::before{
  content:""; }

.bp3-icon-underline::before{
  content:"⎁"; }

.bp3-icon-undo::before{
  content:"⎌"; }

.bp3-icon-ungroup-objects::before{
  content:""; }

.bp3-icon-unknown-vehicle::before{
  content:""; }

.bp3-icon-unlock::before{
  content:""; }

.bp3-icon-unpin::before{
  content:""; }

.bp3-icon-unresolve::before{
  content:""; }

.bp3-icon-updated::before{
  content:""; }

.bp3-icon-upload::before{
  content:""; }

.bp3-icon-user::before{
  content:""; }

.bp3-icon-variable::before{
  content:""; }

.bp3-icon-vertical-bar-chart-asc::before{
  content:""; }

.bp3-icon-vertical-bar-chart-desc::before{
  content:""; }

.bp3-icon-vertical-distribution::before{
  content:""; }

.bp3-icon-video::before{
  content:""; }

.bp3-icon-volume-down::before{
  content:""; }

.bp3-icon-volume-off::before{
  content:""; }

.bp3-icon-volume-up::before{
  content:""; }

.bp3-icon-walk::before{
  content:""; }

.bp3-icon-warning-sign::before{
  content:""; }

.bp3-icon-waterfall-chart::before{
  content:""; }

.bp3-icon-widget::before{
  content:""; }

.bp3-icon-widget-button::before{
  content:""; }

.bp3-icon-widget-footer::before{
  content:""; }

.bp3-icon-widget-header::before{
  content:""; }

.bp3-icon-wrench::before{
  content:""; }

.bp3-icon-zoom-in::before{
  content:""; }

.bp3-icon-zoom-out::before{
  content:""; }

.bp3-icon-zoom-to-fit::before{
  content:""; }
.bp3-submenu > .bp3-popover-wrapper{
  display:block; }

.bp3-submenu .bp3-popover-target{
  display:block; }
  .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{ }

.bp3-submenu.bp3-popover{
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0 5px; }
  .bp3-submenu.bp3-popover > .bp3-popover-content{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-submenu.bp3-popover, .bp3-submenu.bp3-popover.bp3-dark{
    -webkit-box-shadow:none;
            box-shadow:none; }
    .bp3-dark .bp3-submenu.bp3-popover > .bp3-popover-content, .bp3-submenu.bp3-popover.bp3-dark > .bp3-popover-content{
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
.bp3-menu{
  background:#ffffff;
  border-radius:3px;
  color:#182026;
  list-style:none;
  margin:0;
  min-width:180px;
  padding:5px;
  text-align:left; }

.bp3-menu-divider{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px; }
  .bp3-dark .bp3-menu-divider{
    border-top-color:rgba(255, 255, 255, 0.15); }

.bp3-menu-item{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  border-radius:2px;
  color:inherit;
  line-height:20px;
  padding:5px 7px;
  text-decoration:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-menu-item > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-menu-item > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-menu-item::before,
  .bp3-menu-item > *{
    margin-right:7px; }
  .bp3-menu-item:empty::before,
  .bp3-menu-item > :last-child{
    margin-right:0; }
  .bp3-menu-item > .bp3-fill{
    word-break:break-word; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    background-color:rgba(167, 182, 194, 0.3);
    cursor:pointer;
    text-decoration:none; }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-dark .bp3-menu-item{
    color:inherit; }
    .bp3-dark .bp3-menu-item:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
      background-color:rgba(138, 155, 168, 0.15);
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-disabled{
      background-color:inherit;
      color:rgba(167, 182, 194, 0.6); }
  .bp3-menu-item.bp3-intent-primary{
    color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-primary::before, .bp3-menu-item.bp3-intent-primary::after,
    .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-menu-item.bp3-intent-primary:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary:active, .bp3-menu-item.bp3-intent-primary:active::before, .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-success{
    color:#0d8050; }
    .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-success::before, .bp3-menu-item.bp3-intent-success::after,
    .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-menu-item.bp3-intent-success:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-menu-item.bp3-intent-success:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-menu-item.bp3-intent-success:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success:active, .bp3-menu-item.bp3-intent-success:active::before, .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-warning{
    color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-warning::before, .bp3-menu-item.bp3-intent-warning::after,
    .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-menu-item.bp3-intent-warning:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning:active, .bp3-menu-item.bp3-intent-warning:active::before, .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item.bp3-intent-danger{
    color:#c23030; }
    .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-menu-item.bp3-intent-danger::before, .bp3-menu-item.bp3-intent-danger::after,
    .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-menu-item.bp3-intent-danger:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger:active, .bp3-menu-item.bp3-intent-danger:active::before, .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-menu-item::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    margin-right:7px; }
  .bp3-menu-item::before,
  .bp3-menu-item > .bp3-icon{
    color:#5c7080;
    margin-top:2px; }
  .bp3-menu-item .bp3-menu-item-label{
    color:#5c7080; }
  .bp3-menu-item:hover, .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-menu-item{
    color:inherit; }
  .bp3-menu-item.bp3-active, .bp3-menu-item:active{
    background-color:rgba(115, 134, 148, 0.3); }
  .bp3-menu-item.bp3-disabled{
    background-color:inherit !important;
    color:rgba(92, 112, 128, 0.6) !important;
    cursor:not-allowed !important;
    outline:none !important; }
    .bp3-menu-item.bp3-disabled::before,
    .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(92, 112, 128, 0.6) !important; }
  .bp3-large .bp3-menu-item{
    font-size:16px;
    line-height:22px;
    padding:9px 7px; }
    .bp3-large .bp3-menu-item .bp3-icon{
      margin-top:3px; }
    .bp3-large .bp3-menu-item::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1;
      -moz-osx-font-smoothing:grayscale;
      -webkit-font-smoothing:antialiased;
      margin-right:10px;
      margin-top:1px; }

button.bp3-menu-item{
  background:none;
  border:none;
  text-align:left;
  width:100%; }
.bp3-menu-header{
  border-top:1px solid rgba(16, 22, 26, 0.15);
  display:block;
  margin:5px;
  cursor:default;
  padding-left:2px; }
  .bp3-dark .bp3-menu-header{
    border-top-color:rgba(255, 255, 255, 0.15); }
  .bp3-menu-header:first-of-type{
    border-top:none; }
  .bp3-menu-header > h6{
    color:#182026;
    font-weight:600;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    word-wrap:normal;
    line-height:17px;
    margin:0;
    padding:10px 7px 0 1px; }
    .bp3-dark .bp3-menu-header > h6{
      color:#f5f8fa; }
  .bp3-menu-header:first-of-type > h6{
    padding-top:0; }
  .bp3-large .bp3-menu-header > h6{
    font-size:18px;
    padding-bottom:5px;
    padding-top:15px; }
  .bp3-large .bp3-menu-header:first-of-type > h6{
    padding-top:0; }

.bp3-dark .bp3-menu{
  background:#30404d;
  color:#f5f8fa; }

.bp3-dark .bp3-menu-item{ }
  .bp3-dark .bp3-menu-item.bp3-intent-primary{
    color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary::before, .bp3-dark .bp3-menu-item.bp3-intent-primary::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary .bp3-menu-item-label{
      color:#48aff0; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active{
      background-color:#137cbd; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active{
      background-color:#106ba3; }
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-primary.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary:active, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-primary.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-success{
    color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-success::before, .bp3-dark .bp3-menu-item.bp3-intent-success::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success .bp3-menu-item-label{
      color:#3dcc91; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active{
      background-color:#0f9960; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:active{
      background-color:#0d8050; }
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-success:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-success.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success:active, .bp3-dark .bp3-menu-item.bp3-intent-success:active::before, .bp3-dark .bp3-menu-item.bp3-intent-success:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-success.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-warning{
    color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning::before, .bp3-dark .bp3-menu-item.bp3-intent-warning::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning .bp3-menu-item-label{
      color:#ffb366; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active{
      background-color:#d9822b; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active{
      background-color:#bf7326; }
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-warning.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning:active, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-warning.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item.bp3-intent-danger{
    color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-icon{
      color:inherit; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger::before, .bp3-dark .bp3-menu-item.bp3-intent-danger::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger .bp3-menu-item-label{
      color:#ff7373; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active{
      background-color:#db3737; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active{
      background-color:#c23030; }
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::before, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:hover::after, .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after, .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:hover .bp3-menu-item-label,
    .bp3-dark .bp3-submenu .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label,
    .bp3-submenu .bp3-dark .bp3-popover-target.bp3-popover-open > .bp3-intent-danger.bp3-menu-item .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger:active, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger:active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger:active .bp3-menu-item-label, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::before, .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active::after,
    .bp3-dark .bp3-menu-item.bp3-intent-danger.bp3-active .bp3-menu-item-label{
      color:#ffffff; }
  .bp3-dark .bp3-menu-item::before,
  .bp3-dark .bp3-menu-item > .bp3-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item .bp3-menu-item-label{
    color:#a7b6c2; }
  .bp3-dark .bp3-menu-item.bp3-active, .bp3-dark .bp3-menu-item:active{
    background-color:rgba(138, 155, 168, 0.3); }
  .bp3-dark .bp3-menu-item.bp3-disabled{
    color:rgba(167, 182, 194, 0.6) !important; }
    .bp3-dark .bp3-menu-item.bp3-disabled::before,
    .bp3-dark .bp3-menu-item.bp3-disabled > .bp3-icon,
    .bp3-dark .bp3-menu-item.bp3-disabled .bp3-menu-item-label{
      color:rgba(167, 182, 194, 0.6) !important; }

.bp3-dark .bp3-menu-divider,
.bp3-dark .bp3-menu-header{
  border-color:rgba(255, 255, 255, 0.15); }

.bp3-dark .bp3-menu-header > h6{
  color:#f5f8fa; }

.bp3-label .bp3-menu{
  margin-top:5px; }
.bp3-navbar{
  background-color:#ffffff;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.2);
  height:50px;
  padding:0 15px;
  position:relative;
  width:100%;
  z-index:10; }
  .bp3-navbar.bp3-dark,
  .bp3-dark .bp3-navbar{
    background-color:#394b59; }
  .bp3-navbar.bp3-dark{
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-dark .bp3-navbar{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 0 0 rgba(16, 22, 26, 0), 0 1px 1px rgba(16, 22, 26, 0.4); }
  .bp3-navbar.bp3-fixed-top{
    left:0;
    position:fixed;
    right:0;
    top:0; }

.bp3-navbar-heading{
  font-size:16px;
  margin-right:15px; }

.bp3-navbar-group{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:50px; }
  .bp3-navbar-group.bp3-align-left{
    float:left; }
  .bp3-navbar-group.bp3-align-right{
    float:right; }

.bp3-navbar-divider{
  border-left:1px solid rgba(16, 22, 26, 0.15);
  height:20px;
  margin:0 10px; }
  .bp3-dark .bp3-navbar-divider{
    border-left-color:rgba(255, 255, 255, 0.15); }
.bp3-non-ideal-state{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  height:100%;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  text-align:center;
  width:100%; }
  .bp3-non-ideal-state > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-non-ideal-state > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-non-ideal-state::before,
  .bp3-non-ideal-state > *{
    margin-bottom:20px; }
  .bp3-non-ideal-state:empty::before,
  .bp3-non-ideal-state > :last-child{
    margin-bottom:0; }
  .bp3-non-ideal-state > *{
    max-width:400px; }

.bp3-non-ideal-state-visual{
  color:rgba(92, 112, 128, 0.6);
  font-size:60px; }
  .bp3-dark .bp3-non-ideal-state-visual{
    color:rgba(167, 182, 194, 0.6); }

.bp3-overflow-list{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-wrap:nowrap;
      flex-wrap:nowrap;
  min-width:0; }

.bp3-overflow-list-spacer{
  -ms-flex-negative:1;
      flex-shrink:1;
  width:1px; }

body.bp3-overlay-open{
  overflow:hidden; }

.bp3-overlay{
  bottom:0;
  left:0;
  position:static;
  right:0;
  top:0;
  z-index:20; }
  .bp3-overlay:not(.bp3-overlay-open){
    pointer-events:none; }
  .bp3-overlay.bp3-overlay-container{
    overflow:hidden;
    position:fixed; }
    .bp3-overlay.bp3-overlay-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-scroll-container{
    overflow:auto;
    position:fixed; }
    .bp3-overlay.bp3-overlay-scroll-container.bp3-overlay-inline{
      position:absolute; }
  .bp3-overlay.bp3-overlay-inline{
    display:inline;
    overflow:visible; }

.bp3-overlay-content{
  position:fixed;
  z-index:20; }
  .bp3-overlay-inline .bp3-overlay-content,
  .bp3-overlay-scroll-container .bp3-overlay-content{
    position:absolute; }

.bp3-overlay-backdrop{
  bottom:0;
  left:0;
  position:fixed;
  right:0;
  top:0;
  opacity:1;
  background-color:rgba(16, 22, 26, 0.7);
  overflow:auto;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none;
  z-index:20; }
  .bp3-overlay-backdrop.bp3-overlay-enter, .bp3-overlay-backdrop.bp3-overlay-appear{
    opacity:0; }
  .bp3-overlay-backdrop.bp3-overlay-enter-active, .bp3-overlay-backdrop.bp3-overlay-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop.bp3-overlay-exit{
    opacity:1; }
  .bp3-overlay-backdrop.bp3-overlay-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-overlay-backdrop:focus{
    outline:none; }
  .bp3-overlay-inline .bp3-overlay-backdrop{
    position:absolute; }
.bp3-panel-stack{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  height:30px;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack-view{
  bottom:0;
  left:0;
  position:absolute;
  right:0;
  top:0;
  background-color:#ffffff;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  overflow-y:auto;
  z-index:1; }
  .bp3-dark .bp3-panel-stack-view{
    background-color:#30404d; }
  .bp3-panel-stack-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack-push .bp3-panel-stack-enter, .bp3-panel-stack-push .bp3-panel-stack-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack-push .bp3-panel-stack-enter-active, .bp3-panel-stack-push .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-push .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-push .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-enter, .bp3-panel-stack-pop .bp3-panel-stack-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack-pop .bp3-panel-stack-enter-active, .bp3-panel-stack-pop .bp3-panel-stack-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack-pop .bp3-panel-stack-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack-pop .bp3-panel-stack-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }
.bp3-panel-stack2{
  overflow:hidden;
  position:relative; }

.bp3-panel-stack2-header{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  -webkit-box-shadow:0 1px rgba(16, 22, 26, 0.15);
          box-shadow:0 1px rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -ms-flex-negative:0;
      flex-shrink:0;
  height:30px;
  z-index:1; }
  .bp3-dark .bp3-panel-stack2-header{
    -webkit-box-shadow:0 1px rgba(255, 255, 255, 0.15);
            box-shadow:0 1px rgba(255, 255, 255, 0.15); }
  .bp3-panel-stack2-header > span{
    -webkit-box-align:stretch;
        -ms-flex-align:stretch;
            align-items:stretch;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-flex:1;
        -ms-flex:1;
            flex:1; }
  .bp3-panel-stack2-header .bp3-heading{
    margin:0 5px; }

.bp3-button.bp3-panel-stack2-header-back{
  margin-left:5px;
  padding-left:0;
  white-space:nowrap; }
  .bp3-button.bp3-panel-stack2-header-back .bp3-icon{
    margin:0 2px; }

.bp3-panel-stack2-view{
  bottom:0;
  left:0;
  position:absolute;
  right:0;
  top:0;
  background-color:#ffffff;
  border-right:1px solid rgba(16, 22, 26, 0.15);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  margin-right:-1px;
  overflow-y:auto;
  z-index:1; }
  .bp3-dark .bp3-panel-stack2-view{
    background-color:#30404d; }
  .bp3-panel-stack2-view:nth-last-child(n + 4){
    display:none; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter, .bp3-panel-stack2-push .bp3-panel-stack2-appear{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0; }

.bp3-panel-stack2-push .bp3-panel-stack2-enter-active, .bp3-panel-stack2-push .bp3-panel-stack2-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-push .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-push .bp3-panel-stack2-exit-active{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-pop .bp3-panel-stack2-enter, .bp3-panel-stack2-pop .bp3-panel-stack2-appear{
  -webkit-transform:translateX(-50%);
          transform:translateX(-50%);
  opacity:0; }

.bp3-panel-stack2-pop .bp3-panel-stack2-enter-active, .bp3-panel-stack2-pop .bp3-panel-stack2-appear-active{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }

.bp3-panel-stack2-pop .bp3-panel-stack2-exit{
  -webkit-transform:translate(0%);
          transform:translate(0%);
  opacity:1; }

.bp3-panel-stack2-pop .bp3-panel-stack2-exit-active{
  -webkit-transform:translateX(100%);
          transform:translateX(100%);
  opacity:0;
  -webkit-transition-delay:0;
          transition-delay:0;
  -webkit-transition-duration:400ms;
          transition-duration:400ms;
  -webkit-transition-property:opacity, -webkit-transform;
  transition-property:opacity, -webkit-transform;
  transition-property:transform, opacity;
  transition-property:transform, opacity, -webkit-transform;
  -webkit-transition-timing-function:ease;
          transition-timing-function:ease; }
.bp3-popover{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1);
  border-radius:3px;
  display:inline-block;
  z-index:20; }
  .bp3-popover .bp3-popover-arrow{
    height:30px;
    position:absolute;
    width:30px; }
    .bp3-popover .bp3-popover-arrow::before{
      height:20px;
      margin:5px;
      width:20px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover{
    margin-bottom:17px;
    margin-top:-17px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
      bottom:-11px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover{
    margin-left:17px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
      left:-11px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover{
    margin-top:17px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
      top:-11px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover{
    margin-left:-17px;
    margin-right:17px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
      right:-11px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-popover > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-popover > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-popover > .bp3-popover-arrow{
    top:-0.3934px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-popover > .bp3-popover-arrow{
    right:-0.3934px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-popover > .bp3-popover-arrow{
    left:-0.3934px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-popover > .bp3-popover-arrow{
    bottom:-0.3934px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-popover{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-popover{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-popover{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-popover .bp3-popover-content{
    background:#ffffff;
    color:inherit; }
  .bp3-popover .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-popover .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-popover .bp3-popover-arrow-fill{
    fill:#ffffff; }
  .bp3-popover-enter > .bp3-popover, .bp3-popover-appear > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3); }
  .bp3-popover-enter-active > .bp3-popover, .bp3-popover-appear-active > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover-exit > .bp3-popover{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-popover{
    -webkit-transform:scale(0.3);
            transform:scale(0.3);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-popover .bp3-popover-content{
    border-radius:3px;
    position:relative; }
  .bp3-popover.bp3-popover-content-sizing .bp3-popover-content{
    max-width:350px;
    padding:20px; }
  .bp3-popover-target + .bp3-overlay .bp3-popover.bp3-popover-content-sizing{
    width:350px; }
  .bp3-popover.bp3-minimal{
    margin:0 !important; }
    .bp3-popover.bp3-minimal .bp3-popover-arrow{
      display:none; }
    .bp3-popover.bp3-minimal.bp3-popover{
      -webkit-transform:scale(1);
              transform:scale(1); }
      .bp3-popover-enter > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-enter-active > .bp3-popover.bp3-minimal.bp3-popover, .bp3-popover-appear-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
      .bp3-popover-exit > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1); }
      .bp3-popover-exit-active > .bp3-popover.bp3-minimal.bp3-popover{
        -webkit-transform:scale(1);
                transform:scale(1);
        -webkit-transition-delay:0;
                transition-delay:0;
        -webkit-transition-duration:100ms;
                transition-duration:100ms;
        -webkit-transition-property:-webkit-transform;
        transition-property:-webkit-transform;
        transition-property:transform;
        transition-property:transform, -webkit-transform;
        -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
                transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover.bp3-dark,
  .bp3-dark .bp3-popover{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-popover .bp3-popover-content{
      background:#30404d;
      color:inherit; }
    .bp3-popover.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-popover .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-popover.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-popover .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-popover.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-popover .bp3-popover-arrow-fill{
      fill:#30404d; }

.bp3-popover-arrow::before{
  border-radius:2px;
  content:"";
  display:block;
  position:absolute;
  -webkit-transform:rotate(45deg);
          transform:rotate(45deg); }

.bp3-tether-pinned .bp3-popover-arrow{
  display:none; }

.bp3-popover-backdrop{
  background:rgba(255, 255, 255, 0); }

.bp3-transition-container{
  opacity:1;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  z-index:20; }
  .bp3-transition-container.bp3-popover-enter, .bp3-transition-container.bp3-popover-appear{
    opacity:0; }
  .bp3-transition-container.bp3-popover-enter-active, .bp3-transition-container.bp3-popover-appear-active{
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container.bp3-popover-exit{
    opacity:1; }
  .bp3-transition-container.bp3-popover-exit-active{
    opacity:0;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:opacity;
    transition-property:opacity;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-transition-container:focus{
    outline:none; }
  .bp3-transition-container.bp3-popover-leave .bp3-popover-content{
    pointer-events:none; }
  .bp3-transition-container[data-x-out-of-boundaries]{
    display:none; }

span.bp3-popover-target{
  display:inline-block; }

.bp3-popover-wrapper.bp3-fill{
  width:100%; }

.bp3-portal{
  left:0;
  position:absolute;
  right:0;
  top:0; }
@-webkit-keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }
@keyframes linear-progress-bar-stripes{
  from{
    background-position:0 0; }
  to{
    background-position:30px 0; } }

.bp3-progress-bar{
  background:rgba(92, 112, 128, 0.2);
  border-radius:40px;
  display:block;
  height:8px;
  overflow:hidden;
  position:relative;
  width:100%; }
  .bp3-progress-bar .bp3-progress-meter{
    background:linear-gradient(-45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%);
    background-color:rgba(92, 112, 128, 0.8);
    background-size:30px 30px;
    border-radius:40px;
    height:100%;
    position:absolute;
    -webkit-transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:width 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    width:100%; }
  .bp3-progress-bar:not(.bp3-no-animation):not(.bp3-no-stripes) .bp3-progress-meter{
    animation:linear-progress-bar-stripes 300ms linear infinite reverse; }
  .bp3-progress-bar.bp3-no-stripes .bp3-progress-meter{
    background-image:none; }

.bp3-dark .bp3-progress-bar{
  background:rgba(16, 22, 26, 0.5); }
  .bp3-dark .bp3-progress-bar .bp3-progress-meter{
    background-color:#8a9ba8; }

.bp3-progress-bar.bp3-intent-primary .bp3-progress-meter{
  background-color:#137cbd; }

.bp3-progress-bar.bp3-intent-success .bp3-progress-meter{
  background-color:#0f9960; }

.bp3-progress-bar.bp3-intent-warning .bp3-progress-meter{
  background-color:#d9822b; }

.bp3-progress-bar.bp3-intent-danger .bp3-progress-meter{
  background-color:#db3737; }
@-webkit-keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
@keyframes skeleton-glow{
  from{
    background:rgba(206, 217, 224, 0.2);
    border-color:rgba(206, 217, 224, 0.2); }
  to{
    background:rgba(92, 112, 128, 0.2);
    border-color:rgba(92, 112, 128, 0.2); } }
.bp3-skeleton{
  -webkit-animation:1000ms linear infinite alternate skeleton-glow;
          animation:1000ms linear infinite alternate skeleton-glow;
  background:rgba(206, 217, 224, 0.2);
  background-clip:padding-box !important;
  border-color:rgba(206, 217, 224, 0.2) !important;
  border-radius:2px;
  -webkit-box-shadow:none !important;
          box-shadow:none !important;
  color:transparent !important;
  cursor:default;
  pointer-events:none;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-skeleton::before, .bp3-skeleton::after,
  .bp3-skeleton *{
    visibility:hidden !important; }
.bp3-slider{
  height:40px;
  min-width:150px;
  width:100%;
  cursor:default;
  outline:none;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-slider:hover{
    cursor:pointer; }
  .bp3-slider:active{
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-slider.bp3-disabled{
    cursor:not-allowed;
    opacity:0.5; }
  .bp3-slider.bp3-slider-unlabeled{
    height:16px; }

.bp3-slider-track,
.bp3-slider-progress{
  height:6px;
  left:0;
  right:0;
  top:5px;
  position:absolute; }

.bp3-slider-track{
  border-radius:3px;
  overflow:hidden; }

.bp3-slider-progress{
  background:rgba(92, 112, 128, 0.2); }
  .bp3-dark .bp3-slider-progress{
    background:rgba(16, 22, 26, 0.5); }
  .bp3-slider-progress.bp3-intent-primary{
    background-color:#137cbd; }
  .bp3-slider-progress.bp3-intent-success{
    background-color:#0f9960; }
  .bp3-slider-progress.bp3-intent-warning{
    background-color:#d9822b; }
  .bp3-slider-progress.bp3-intent-danger{
    background-color:#db3737; }

.bp3-slider-handle{
  background-color:#f5f8fa;
  background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.8)), to(rgba(255, 255, 255, 0)));
  background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0));
  -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
          box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
  color:#182026;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
  cursor:pointer;
  height:16px;
  left:0;
  position:absolute;
  top:0;
  width:16px; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1); }
  .bp3-slider-handle:active, .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
  .bp3-slider-handle:disabled, .bp3-slider-handle.bp3-disabled{
    background-color:rgba(206, 217, 224, 0.5);
    background-image:none;
    -webkit-box-shadow:none;
            box-shadow:none;
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed;
    outline:none; }
    .bp3-slider-handle:disabled.bp3-active, .bp3-slider-handle:disabled.bp3-active:hover, .bp3-slider-handle.bp3-disabled.bp3-active, .bp3-slider-handle.bp3-disabled.bp3-active:hover{
      background:rgba(206, 217, 224, 0.7); }
  .bp3-slider-handle:focus{
    z-index:1; }
  .bp3-slider-handle:hover{
    background-clip:padding-box;
    background-color:#ebf1f5;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 -1px 0 rgba(16, 22, 26, 0.1);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 1px 1px rgba(16, 22, 26, 0.2);
    cursor:-webkit-grab;
    cursor:grab;
    z-index:2; }
  .bp3-slider-handle.bp3-active{
    background-color:#d8e1e8;
    background-image:none;
    -webkit-box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
            box-shadow:inset 0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 2px rgba(16, 22, 26, 0.2);
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), inset 0 1px 1px rgba(16, 22, 26, 0.1);
    cursor:-webkit-grabbing;
    cursor:grabbing; }
  .bp3-disabled .bp3-slider-handle{
    background:#bfccd6;
    -webkit-box-shadow:none;
            box-shadow:none;
    pointer-events:none; }
  .bp3-dark .bp3-slider-handle{
    background-color:#394b59;
    background-image:-webkit-gradient(linear, left top, left bottom, from(rgba(255, 255, 255, 0.05)), to(rgba(255, 255, 255, 0)));
    background-image:linear-gradient(to bottom, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0));
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
    color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover, .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      color:#f5f8fa; }
    .bp3-dark .bp3-slider-handle:hover{
      background-color:#30404d;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-slider-handle:active, .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#202b33;
      background-image:none;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.6), inset 0 1px 2px rgba(16, 22, 26, 0.2); }
    .bp3-dark .bp3-slider-handle:disabled, .bp3-dark .bp3-slider-handle.bp3-disabled{
      background-color:rgba(57, 75, 89, 0.5);
      background-image:none;
      -webkit-box-shadow:none;
              box-shadow:none;
      color:rgba(167, 182, 194, 0.6); }
      .bp3-dark .bp3-slider-handle:disabled.bp3-active, .bp3-dark .bp3-slider-handle.bp3-disabled.bp3-active{
        background:rgba(57, 75, 89, 0.7); }
    .bp3-dark .bp3-slider-handle .bp3-button-spinner .bp3-spinner-head{
      background:rgba(16, 22, 26, 0.5);
      stroke:#8a9ba8; }
    .bp3-dark .bp3-slider-handle, .bp3-dark .bp3-slider-handle:hover{
      background-color:#394b59; }
    .bp3-dark .bp3-slider-handle.bp3-active{
      background-color:#293742; }
  .bp3-dark .bp3-disabled .bp3-slider-handle{
    background:#5c7080;
    border-color:#5c7080;
    -webkit-box-shadow:none;
            box-shadow:none; }
  .bp3-slider-handle .bp3-slider-label{
    background:#394b59;
    border-radius:3px;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
    color:#f5f8fa;
    margin-left:8px; }
    .bp3-dark .bp3-slider-handle .bp3-slider-label{
      background:#e1e8ed;
      -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
      color:#394b59; }
    .bp3-disabled .bp3-slider-handle .bp3-slider-label{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-slider-handle.bp3-start, .bp3-slider-handle.bp3-end{
    width:8px; }
  .bp3-slider-handle.bp3-start{
    border-bottom-right-radius:0;
    border-top-right-radius:0; }
  .bp3-slider-handle.bp3-end{
    border-bottom-left-radius:0;
    border-top-left-radius:0;
    margin-left:8px; }
    .bp3-slider-handle.bp3-end .bp3-slider-label{
      margin-left:0; }

.bp3-slider-label{
  -webkit-transform:translate(-50%, 20px);
          transform:translate(-50%, 20px);
  display:inline-block;
  font-size:12px;
  line-height:1;
  padding:2px 5px;
  position:absolute;
  vertical-align:top; }

.bp3-slider.bp3-vertical{
  height:150px;
  min-width:40px;
  width:40px; }
  .bp3-slider.bp3-vertical .bp3-slider-track,
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    bottom:0;
    height:auto;
    left:5px;
    top:0;
    width:6px; }
  .bp3-slider.bp3-vertical .bp3-slider-progress{
    top:auto; }
  .bp3-slider.bp3-vertical .bp3-slider-label{
    -webkit-transform:translate(20px, 50%);
            transform:translate(20px, 50%); }
  .bp3-slider.bp3-vertical .bp3-slider-handle{
    top:auto; }
    .bp3-slider.bp3-vertical .bp3-slider-handle .bp3-slider-label{
      margin-left:0;
      margin-top:-8px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end, .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      height:8px;
      margin-left:0;
      width:16px; }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start{
      border-bottom-right-radius:3px;
      border-top-left-radius:0; }
      .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-start .bp3-slider-label{
        -webkit-transform:translate(20px);
                transform:translate(20px); }
    .bp3-slider.bp3-vertical .bp3-slider-handle.bp3-end{
      border-bottom-left-radius:0;
      border-bottom-right-radius:0;
      border-top-left-radius:3px;
      margin-bottom:8px; }

@-webkit-keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

@keyframes pt-spinner-animation{
  from{
    -webkit-transform:rotate(0deg);
            transform:rotate(0deg); }
  to{
    -webkit-transform:rotate(360deg);
            transform:rotate(360deg); } }

.bp3-spinner{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-pack:center;
      -ms-flex-pack:center;
          justify-content:center;
  overflow:visible;
  vertical-align:middle; }
  .bp3-spinner svg{
    display:block; }
  .bp3-spinner path{
    fill-opacity:0; }
  .bp3-spinner .bp3-spinner-head{
    stroke:rgba(92, 112, 128, 0.8);
    stroke-linecap:round;
    -webkit-transform-origin:center;
            transform-origin:center;
    -webkit-transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
    transition:stroke-dashoffset 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-spinner .bp3-spinner-track{
    stroke:rgba(92, 112, 128, 0.2); }

.bp3-spinner-animation{
  -webkit-animation:pt-spinner-animation 500ms linear infinite;
          animation:pt-spinner-animation 500ms linear infinite; }
  .bp3-no-spin > .bp3-spinner-animation{
    -webkit-animation:none;
            animation:none; }

.bp3-dark .bp3-spinner .bp3-spinner-head{
  stroke:#8a9ba8; }

.bp3-dark .bp3-spinner .bp3-spinner-track{
  stroke:rgba(16, 22, 26, 0.5); }

.bp3-spinner.bp3-intent-primary .bp3-spinner-head{
  stroke:#137cbd; }

.bp3-spinner.bp3-intent-success .bp3-spinner-head{
  stroke:#0f9960; }

.bp3-spinner.bp3-intent-warning .bp3-spinner-head{
  stroke:#d9822b; }

.bp3-spinner.bp3-intent-danger .bp3-spinner-head{
  stroke:#db3737; }
.bp3-tabs.bp3-vertical{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex; }
  .bp3-tabs.bp3-vertical > .bp3-tab-list{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start;
    -webkit-box-orient:vertical;
    -webkit-box-direction:normal;
        -ms-flex-direction:column;
            flex-direction:column; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab{
      border-radius:3px;
      padding:0 10px;
      width:100%; }
      .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab[aria-selected="true"]{
        background-color:rgba(19, 124, 189, 0.2);
        -webkit-box-shadow:none;
                box-shadow:none; }
    .bp3-tabs.bp3-vertical > .bp3-tab-list .bp3-tab-indicator-wrapper .bp3-tab-indicator{
      background-color:rgba(19, 124, 189, 0.2);
      border-radius:3px;
      bottom:0;
      height:auto;
      left:0;
      right:0;
      top:0; }
  .bp3-tabs.bp3-vertical > .bp3-tab-panel{
    margin-top:0;
    padding-left:20px; }

.bp3-tab-list{
  -webkit-box-align:end;
      -ms-flex-align:end;
          align-items:flex-end;
  border:none;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  list-style:none;
  margin:0;
  padding:0;
  position:relative; }
  .bp3-tab-list > *:not(:last-child){
    margin-right:20px; }

.bp3-tab{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  color:#182026;
  cursor:pointer;
  -webkit-box-flex:0;
      -ms-flex:0 0 auto;
          flex:0 0 auto;
  font-size:14px;
  line-height:30px;
  max-width:100%;
  position:relative;
  vertical-align:top; }
  .bp3-tab a{
    color:inherit;
    display:block;
    text-decoration:none; }
  .bp3-tab-indicator-wrapper ~ .bp3-tab{
    background-color:transparent !important;
    -webkit-box-shadow:none !important;
            box-shadow:none !important; }
  .bp3-tab[aria-disabled="true"]{
    color:rgba(92, 112, 128, 0.6);
    cursor:not-allowed; }
  .bp3-tab[aria-selected="true"]{
    border-radius:0;
    -webkit-box-shadow:inset 0 -3px 0 #106ba3;
            box-shadow:inset 0 -3px 0 #106ba3; }
  .bp3-tab[aria-selected="true"], .bp3-tab:not([aria-disabled="true"]):hover{
    color:#106ba3; }
  .bp3-tab:focus{
    -moz-outline-radius:0; }
  .bp3-large > .bp3-tab{
    font-size:16px;
    line-height:40px; }

.bp3-tab-panel{
  margin-top:20px; }
  .bp3-tab-panel[aria-hidden="true"]{
    display:none; }

.bp3-tab-indicator-wrapper{
  left:0;
  pointer-events:none;
  position:absolute;
  top:0;
  -webkit-transform:translateX(0), translateY(0);
          transform:translateX(0), translateY(0);
  -webkit-transition:height, width, -webkit-transform;
  transition:height, width, -webkit-transform;
  transition:height, transform, width;
  transition:height, transform, width, -webkit-transform;
  -webkit-transition-duration:200ms;
          transition-duration:200ms;
  -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
          transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tab-indicator-wrapper .bp3-tab-indicator{
    background-color:#106ba3;
    bottom:0;
    height:3px;
    left:0;
    position:absolute;
    right:0; }
  .bp3-tab-indicator-wrapper.bp3-no-animation{
    -webkit-transition:none;
    transition:none; }

.bp3-dark .bp3-tab{
  color:#f5f8fa; }
  .bp3-dark .bp3-tab[aria-disabled="true"]{
    color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tab[aria-selected="true"]{
    -webkit-box-shadow:inset 0 -3px 0 #48aff0;
            box-shadow:inset 0 -3px 0 #48aff0; }
  .bp3-dark .bp3-tab[aria-selected="true"], .bp3-dark .bp3-tab:not([aria-disabled="true"]):hover{
    color:#48aff0; }

.bp3-dark .bp3-tab-indicator{
  background-color:#48aff0; }

.bp3-flex-expander{
  -webkit-box-flex:1;
      -ms-flex:1 1;
          flex:1 1; }
.bp3-tag{
  display:-webkit-inline-box;
  display:-ms-inline-flexbox;
  display:inline-flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  background-color:#5c7080;
  border:none;
  border-radius:3px;
  -webkit-box-shadow:none;
          box-shadow:none;
  color:#f5f8fa;
  font-size:12px;
  line-height:16px;
  max-width:100%;
  min-height:20px;
  min-width:20px;
  padding:2px 6px;
  position:relative; }
  .bp3-tag.bp3-interactive{
    cursor:pointer; }
    .bp3-tag.bp3-interactive:hover{
      background-color:rgba(92, 112, 128, 0.85); }
    .bp3-tag.bp3-interactive.bp3-active, .bp3-tag.bp3-interactive:active{
      background-color:rgba(92, 112, 128, 0.7); }
  .bp3-tag > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag > .bp3-fill{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag::before,
  .bp3-tag > *{
    margin-right:4px; }
  .bp3-tag:empty::before,
  .bp3-tag > :last-child{
    margin-right:0; }
  .bp3-tag:focus{
    outline:rgba(19, 124, 189, 0.6) auto 2px;
    outline-offset:0;
    -moz-outline-radius:6px; }
  .bp3-tag.bp3-round{
    border-radius:30px;
    padding-left:8px;
    padding-right:8px; }
  .bp3-dark .bp3-tag{
    background-color:#bfccd6;
    color:#182026; }
    .bp3-dark .bp3-tag.bp3-interactive{
      cursor:pointer; }
      .bp3-dark .bp3-tag.bp3-interactive:hover{
        background-color:rgba(191, 204, 214, 0.85); }
      .bp3-dark .bp3-tag.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-interactive:active{
        background-color:rgba(191, 204, 214, 0.7); }
    .bp3-dark .bp3-tag > .bp3-icon, .bp3-dark .bp3-tag .bp3-icon-standard, .bp3-dark .bp3-tag .bp3-icon-large{
      fill:currentColor; }
  .bp3-tag > .bp3-icon, .bp3-tag .bp3-icon-standard, .bp3-tag .bp3-icon-large{
    fill:#ffffff; }
  .bp3-tag.bp3-large,
  .bp3-large .bp3-tag{
    font-size:14px;
    line-height:20px;
    min-height:30px;
    min-width:30px;
    padding:5px 10px; }
    .bp3-tag.bp3-large::before,
    .bp3-tag.bp3-large > *,
    .bp3-large .bp3-tag::before,
    .bp3-large .bp3-tag > *{
      margin-right:7px; }
    .bp3-tag.bp3-large:empty::before,
    .bp3-tag.bp3-large > :last-child,
    .bp3-large .bp3-tag:empty::before,
    .bp3-large .bp3-tag > :last-child{
      margin-right:0; }
    .bp3-tag.bp3-large.bp3-round,
    .bp3-large .bp3-tag.bp3-round{
      padding-left:12px;
      padding-right:12px; }
  .bp3-tag.bp3-intent-primary{
    background:#137cbd;
    color:#ffffff; }
    .bp3-tag.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.85); }
      .bp3-tag.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.7); }
  .bp3-tag.bp3-intent-success{
    background:#0f9960;
    color:#ffffff; }
    .bp3-tag.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.85); }
      .bp3-tag.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.7); }
  .bp3-tag.bp3-intent-warning{
    background:#d9822b;
    color:#ffffff; }
    .bp3-tag.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.85); }
      .bp3-tag.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.7); }
  .bp3-tag.bp3-intent-danger{
    background:#db3737;
    color:#ffffff; }
    .bp3-tag.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.85); }
      .bp3-tag.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.7); }
  .bp3-tag.bp3-fill{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    width:100%; }
  .bp3-tag.bp3-minimal > .bp3-icon, .bp3-tag.bp3-minimal .bp3-icon-standard, .bp3-tag.bp3-minimal .bp3-icon-large{
    fill:#5c7080; }
  .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
    background-color:rgba(138, 155, 168, 0.2);
    color:#182026; }
    .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
        background-color:rgba(92, 112, 128, 0.3); }
      .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
        background-color:rgba(92, 112, 128, 0.4); }
    .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]){
      color:#f5f8fa; }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:hover{
          background-color:rgba(191, 204, 214, 0.3); }
        .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]).bp3-interactive:active{
          background-color:rgba(191, 204, 214, 0.4); }
      .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) > .bp3-icon, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-standard, .bp3-dark .bp3-tag.bp3-minimal:not([class*="bp3-intent-"]) .bp3-icon-large{
        fill:#a7b6c2; }
  .bp3-tag.bp3-minimal.bp3-intent-primary{
    background-color:rgba(19, 124, 189, 0.15);
    color:#106ba3; }
    .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
        background-color:rgba(19, 124, 189, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
        background-color:rgba(19, 124, 189, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-primary > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-primary .bp3-icon-large{
      fill:#137cbd; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary{
      background-color:rgba(19, 124, 189, 0.25);
      color:#48aff0; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:hover{
          background-color:rgba(19, 124, 189, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-primary.bp3-interactive:active{
          background-color:rgba(19, 124, 189, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-success{
    background-color:rgba(15, 153, 96, 0.15);
    color:#0d8050; }
    .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
        background-color:rgba(15, 153, 96, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
        background-color:rgba(15, 153, 96, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-success > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-success .bp3-icon-large{
      fill:#0f9960; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success{
      background-color:rgba(15, 153, 96, 0.25);
      color:#3dcc91; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:hover{
          background-color:rgba(15, 153, 96, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-success.bp3-interactive:active{
          background-color:rgba(15, 153, 96, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-warning{
    background-color:rgba(217, 130, 43, 0.15);
    color:#bf7326; }
    .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
        background-color:rgba(217, 130, 43, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
        background-color:rgba(217, 130, 43, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-warning > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-warning .bp3-icon-large{
      fill:#d9822b; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning{
      background-color:rgba(217, 130, 43, 0.25);
      color:#ffb366; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:hover{
          background-color:rgba(217, 130, 43, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-warning.bp3-interactive:active{
          background-color:rgba(217, 130, 43, 0.45); }
  .bp3-tag.bp3-minimal.bp3-intent-danger{
    background-color:rgba(219, 55, 55, 0.15);
    color:#c23030; }
    .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
      cursor:pointer; }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
        background-color:rgba(219, 55, 55, 0.25); }
      .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
        background-color:rgba(219, 55, 55, 0.35); }
    .bp3-tag.bp3-minimal.bp3-intent-danger > .bp3-icon, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-standard, .bp3-tag.bp3-minimal.bp3-intent-danger .bp3-icon-large{
      fill:#db3737; }
    .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger{
      background-color:rgba(219, 55, 55, 0.25);
      color:#ff7373; }
      .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive{
        cursor:pointer; }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:hover{
          background-color:rgba(219, 55, 55, 0.35); }
        .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive.bp3-active, .bp3-dark .bp3-tag.bp3-minimal.bp3-intent-danger.bp3-interactive:active{
          background-color:rgba(219, 55, 55, 0.45); }

.bp3-tag-remove{
  background:none;
  border:none;
  color:inherit;
  cursor:pointer;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin-bottom:-2px;
  margin-right:-6px !important;
  margin-top:-2px;
  opacity:0.5;
  padding:2px;
  padding-left:0; }
  .bp3-tag-remove:hover{
    background:none;
    opacity:0.8;
    text-decoration:none; }
  .bp3-tag-remove:active{
    opacity:1; }
  .bp3-tag-remove:empty::before{
    font-family:"Icons16", sans-serif;
    font-size:16px;
    font-style:normal;
    font-weight:400;
    line-height:1;
    -moz-osx-font-smoothing:grayscale;
    -webkit-font-smoothing:antialiased;
    content:""; }
  .bp3-large .bp3-tag-remove{
    margin-right:-10px !important;
    padding:0 5px 0 0; }
    .bp3-large .bp3-tag-remove:empty::before{
      font-family:"Icons20", sans-serif;
      font-size:20px;
      font-style:normal;
      font-weight:400;
      line-height:1; }
.bp3-tag-input{
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  -webkit-box-orient:horizontal;
  -webkit-box-direction:normal;
      -ms-flex-direction:row;
          flex-direction:row;
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  cursor:text;
  height:auto;
  line-height:inherit;
  min-height:30px;
  padding-left:5px;
  padding-right:0; }
  .bp3-tag-input > *{
    -webkit-box-flex:0;
        -ms-flex-positive:0;
            flex-grow:0;
    -ms-flex-negative:0;
        flex-shrink:0; }
  .bp3-tag-input > .bp3-tag-input-values{
    -webkit-box-flex:1;
        -ms-flex-positive:1;
            flex-grow:1;
    -ms-flex-negative:1;
        flex-shrink:1; }
  .bp3-tag-input .bp3-tag-input-icon{
    color:#5c7080;
    margin-left:2px;
    margin-right:7px;
    margin-top:7px; }
  .bp3-tag-input .bp3-tag-input-values{
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex;
    -webkit-box-orient:horizontal;
    -webkit-box-direction:normal;
        -ms-flex-direction:row;
            flex-direction:row;
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    -ms-flex-item-align:stretch;
        align-self:stretch;
    -ms-flex-wrap:wrap;
        flex-wrap:wrap;
    margin-right:7px;
    margin-top:5px;
    min-width:0; }
    .bp3-tag-input .bp3-tag-input-values > *{
      -webkit-box-flex:0;
          -ms-flex-positive:0;
              flex-grow:0;
      -ms-flex-negative:0;
          flex-shrink:0; }
    .bp3-tag-input .bp3-tag-input-values > .bp3-fill{
      -webkit-box-flex:1;
          -ms-flex-positive:1;
              flex-grow:1;
      -ms-flex-negative:1;
          flex-shrink:1; }
    .bp3-tag-input .bp3-tag-input-values::before,
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-right:5px; }
    .bp3-tag-input .bp3-tag-input-values:empty::before,
    .bp3-tag-input .bp3-tag-input-values > :last-child{
      margin-right:0; }
    .bp3-tag-input .bp3-tag-input-values:first-child .bp3-input-ghost:first-child{
      padding-left:5px; }
    .bp3-tag-input .bp3-tag-input-values > *{
      margin-bottom:5px; }
  .bp3-tag-input .bp3-tag{
    overflow-wrap:break-word; }
    .bp3-tag-input .bp3-tag.bp3-active{
      outline:rgba(19, 124, 189, 0.6) auto 2px;
      outline-offset:0;
      -moz-outline-radius:6px; }
  .bp3-tag-input .bp3-input-ghost{
    -webkit-box-flex:1;
        -ms-flex:1 1 auto;
            flex:1 1 auto;
    line-height:20px;
    width:80px; }
    .bp3-tag-input .bp3-input-ghost:disabled, .bp3-tag-input .bp3-input-ghost.bp3-disabled{
      cursor:not-allowed; }
  .bp3-tag-input .bp3-button,
  .bp3-tag-input .bp3-spinner{
    margin:3px;
    margin-left:0; }
  .bp3-tag-input .bp3-button{
    min-height:24px;
    min-width:24px;
    padding:0 7px; }
  .bp3-tag-input.bp3-large{
    height:auto;
    min-height:40px; }
    .bp3-tag-input.bp3-large::before,
    .bp3-tag-input.bp3-large > *{
      margin-right:10px; }
    .bp3-tag-input.bp3-large:empty::before,
    .bp3-tag-input.bp3-large > :last-child{
      margin-right:0; }
    .bp3-tag-input.bp3-large .bp3-tag-input-icon{
      margin-left:5px;
      margin-top:10px; }
    .bp3-tag-input.bp3-large .bp3-input-ghost{
      line-height:30px; }
    .bp3-tag-input.bp3-large .bp3-button{
      min-height:30px;
      min-width:30px;
      padding:5px 10px;
      margin:5px;
      margin-left:0; }
    .bp3-tag-input.bp3-large .bp3-spinner{
      margin:8px;
      margin-left:0; }
  .bp3-tag-input.bp3-active{
    background-color:#ffffff;
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
    .bp3-tag-input.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.2); }
  .bp3-dark .bp3-tag-input .bp3-tag-input-icon, .bp3-tag-input.bp3-dark .bp3-tag-input-icon{
    color:#a7b6c2; }
  .bp3-dark .bp3-tag-input .bp3-input-ghost, .bp3-tag-input.bp3-dark .bp3-input-ghost{
    color:#f5f8fa; }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-webkit-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-webkit-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-moz-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-moz-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost:-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost:-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::-ms-input-placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::-ms-input-placeholder{
      color:rgba(167, 182, 194, 0.6); }
    .bp3-dark .bp3-tag-input .bp3-input-ghost::placeholder, .bp3-tag-input.bp3-dark .bp3-input-ghost::placeholder{
      color:rgba(167, 182, 194, 0.6); }
  .bp3-dark .bp3-tag-input.bp3-active, .bp3-tag-input.bp3-dark.bp3-active{
    background-color:rgba(16, 22, 26, 0.3);
    -webkit-box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px #137cbd, 0 0 0 1px #137cbd, 0 0 0 3px rgba(19, 124, 189, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-primary, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-primary{
      -webkit-box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #106ba3, 0 0 0 3px rgba(16, 107, 163, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-success, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-success{
      -webkit-box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #0d8050, 0 0 0 3px rgba(13, 128, 80, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-warning, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-warning{
      -webkit-box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #bf7326, 0 0 0 3px rgba(191, 115, 38, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }
    .bp3-dark .bp3-tag-input.bp3-active.bp3-intent-danger, .bp3-tag-input.bp3-dark.bp3-active.bp3-intent-danger{
      -webkit-box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4);
              box-shadow:0 0 0 1px #c23030, 0 0 0 3px rgba(194, 48, 48, 0.3), inset 0 0 0 1px rgba(16, 22, 26, 0.3), inset 0 1px 1px rgba(16, 22, 26, 0.4); }

.bp3-input-ghost{
  background:none;
  border:none;
  -webkit-box-shadow:none;
          box-shadow:none;
  padding:0; }
  .bp3-input-ghost::-webkit-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-moz-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::-ms-input-placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost::placeholder{
    color:rgba(92, 112, 128, 0.6);
    opacity:1; }
  .bp3-input-ghost:focus{
    outline:none !important; }
.bp3-toast{
  -webkit-box-align:start;
      -ms-flex-align:start;
          align-items:flex-start;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  margin:20px 0 0;
  max-width:500px;
  min-width:300px;
  pointer-events:all;
  position:relative !important; }
  .bp3-toast.bp3-toast-enter, .bp3-toast.bp3-toast-appear{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active, .bp3-toast.bp3-toast-appear-active{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-enter ~ .bp3-toast, .bp3-toast.bp3-toast-appear ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px); }
  .bp3-toast.bp3-toast-enter-active ~ .bp3-toast, .bp3-toast.bp3-toast-appear-active ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11);
            transition-timing-function:cubic-bezier(0.54, 1.12, 0.38, 1.11); }
  .bp3-toast.bp3-toast-exit{
    opacity:1;
    -webkit-filter:blur(0);
            filter:blur(0); }
  .bp3-toast.bp3-toast-exit-active{
    opacity:0;
    -webkit-filter:blur(10px);
            filter:blur(10px);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:300ms;
            transition-duration:300ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:opacity, filter;
    transition-property:opacity, filter, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast.bp3-toast-exit ~ .bp3-toast{
    -webkit-transform:translateY(0);
            transform:translateY(0); }
  .bp3-toast.bp3-toast-exit-active ~ .bp3-toast{
    -webkit-transform:translateY(-40px);
            transform:translateY(-40px);
    -webkit-transition-delay:50ms;
            transition-delay:50ms;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-toast .bp3-button-group{
    -webkit-box-flex:0;
        -ms-flex:0 0 auto;
            flex:0 0 auto;
    padding:5px;
    padding-left:0; }
  .bp3-toast > .bp3-icon{
    color:#5c7080;
    margin:12px;
    margin-right:0; }
  .bp3-toast.bp3-dark,
  .bp3-dark .bp3-toast{
    background-color:#394b59;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-toast.bp3-dark > .bp3-icon,
    .bp3-dark .bp3-toast > .bp3-icon{
      color:#a7b6c2; }
  .bp3-toast[class*="bp3-intent-"] a{
    color:rgba(255, 255, 255, 0.7); }
    .bp3-toast[class*="bp3-intent-"] a:hover{
      color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] > .bp3-icon{
    color:#ffffff; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button, .bp3-toast[class*="bp3-intent-"] .bp3-button::before,
  .bp3-toast[class*="bp3-intent-"] .bp3-button .bp3-icon, .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    color:rgba(255, 255, 255, 0.7) !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:focus{
    outline-color:rgba(255, 255, 255, 0.5); }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:hover{
    background-color:rgba(255, 255, 255, 0.15) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button:active{
    background-color:rgba(255, 255, 255, 0.3) !important;
    color:#ffffff !important; }
  .bp3-toast[class*="bp3-intent-"] .bp3-button::after{
    background:rgba(255, 255, 255, 0.3) !important; }
  .bp3-toast.bp3-intent-primary{
    background-color:#137cbd;
    color:#ffffff; }
  .bp3-toast.bp3-intent-success{
    background-color:#0f9960;
    color:#ffffff; }
  .bp3-toast.bp3-intent-warning{
    background-color:#d9822b;
    color:#ffffff; }
  .bp3-toast.bp3-intent-danger{
    background-color:#db3737;
    color:#ffffff; }

.bp3-toast-message{
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  padding:11px;
  word-break:break-word; }

.bp3-toast-container{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box !important;
  display:-ms-flexbox !important;
  display:flex !important;
  -webkit-box-orient:vertical;
  -webkit-box-direction:normal;
      -ms-flex-direction:column;
          flex-direction:column;
  left:0;
  overflow:hidden;
  padding:0 20px 20px;
  pointer-events:none;
  right:0;
  z-index:40; }
  .bp3-toast-container.bp3-toast-container-in-portal{
    position:fixed; }
  .bp3-toast-container.bp3-toast-container-inline{
    position:absolute; }
  .bp3-toast-container.bp3-toast-container-top{
    top:0; }
  .bp3-toast-container.bp3-toast-container-bottom{
    bottom:0;
    -webkit-box-orient:vertical;
    -webkit-box-direction:reverse;
        -ms-flex-direction:column-reverse;
            flex-direction:column-reverse;
    top:auto; }
  .bp3-toast-container.bp3-toast-container-left{
    -webkit-box-align:start;
        -ms-flex-align:start;
            align-items:flex-start; }
  .bp3-toast-container.bp3-toast-container-right{
    -webkit-box-align:end;
        -ms-flex-align:end;
            align-items:flex-end; }

.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-enter:not(.bp3-toast-enter-active) ~ .bp3-toast, .bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active),
.bp3-toast-container-bottom .bp3-toast.bp3-toast-appear:not(.bp3-toast-appear-active) ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-exit-active ~ .bp3-toast,
.bp3-toast-container-bottom .bp3-toast.bp3-toast-leave-active ~ .bp3-toast{
  -webkit-transform:translateY(60px);
          transform:translateY(60px); }
.bp3-tooltip{
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 2px 4px rgba(16, 22, 26, 0.2), 0 8px 24px rgba(16, 22, 26, 0.2);
  -webkit-transform:scale(1);
          transform:scale(1); }
  .bp3-tooltip .bp3-popover-arrow{
    height:22px;
    position:absolute;
    width:22px; }
    .bp3-tooltip .bp3-popover-arrow::before{
      height:14px;
      margin:4px;
      width:14px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip{
    margin-bottom:11px;
    margin-top:-11px; }
    .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
      bottom:-8px; }
      .bp3-tether-element-attached-bottom.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(-90deg);
                transform:rotate(-90deg); }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip{
    margin-left:11px; }
    .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
      left:-8px; }
      .bp3-tether-element-attached-left.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(0);
                transform:rotate(0); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip{
    margin-top:11px; }
    .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
      top:-8px; }
      .bp3-tether-element-attached-top.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(90deg);
                transform:rotate(90deg); }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip{
    margin-left:-11px;
    margin-right:11px; }
    .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
      right:-8px; }
      .bp3-tether-element-attached-right.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow svg{
        -webkit-transform:rotate(180deg);
                transform:rotate(180deg); }
  .bp3-tether-element-attached-middle > .bp3-tooltip > .bp3-popover-arrow{
    top:50%;
    -webkit-transform:translateY(-50%);
            transform:translateY(-50%); }
  .bp3-tether-element-attached-center > .bp3-tooltip > .bp3-popover-arrow{
    right:50%;
    -webkit-transform:translateX(50%);
            transform:translateX(50%); }
  .bp3-tether-element-attached-top.bp3-tether-target-attached-top > .bp3-tooltip > .bp3-popover-arrow{
    top:-0.22183px; }
  .bp3-tether-element-attached-right.bp3-tether-target-attached-right > .bp3-tooltip > .bp3-popover-arrow{
    right:-0.22183px; }
  .bp3-tether-element-attached-left.bp3-tether-target-attached-left > .bp3-tooltip > .bp3-popover-arrow{
    left:-0.22183px; }
  .bp3-tether-element-attached-bottom.bp3-tether-target-attached-bottom > .bp3-tooltip > .bp3-popover-arrow{
    bottom:-0.22183px; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:top left;
            transform-origin:top left; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:top center;
            transform-origin:top center; }
  .bp3-tether-element-attached-top.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:top right;
            transform-origin:top right; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:center left;
            transform-origin:center left; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:center center;
            transform-origin:center center; }
  .bp3-tether-element-attached-middle.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:center right;
            transform-origin:center right; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-left > .bp3-tooltip{
    -webkit-transform-origin:bottom left;
            transform-origin:bottom left; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-center > .bp3-tooltip{
    -webkit-transform-origin:bottom center;
            transform-origin:bottom center; }
  .bp3-tether-element-attached-bottom.bp3-tether-element-attached-right > .bp3-tooltip{
    -webkit-transform-origin:bottom right;
            transform-origin:bottom right; }
  .bp3-tooltip .bp3-popover-content{
    background:#394b59;
    color:#f5f8fa; }
  .bp3-tooltip .bp3-popover-arrow::before{
    -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2);
            box-shadow:1px 1px 6px rgba(16, 22, 26, 0.2); }
  .bp3-tooltip .bp3-popover-arrow-border{
    fill:#10161a;
    fill-opacity:0.1; }
  .bp3-tooltip .bp3-popover-arrow-fill{
    fill:#394b59; }
  .bp3-popover-enter > .bp3-tooltip, .bp3-popover-appear > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8); }
  .bp3-popover-enter-active > .bp3-tooltip, .bp3-popover-appear-active > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-popover-exit > .bp3-tooltip{
    -webkit-transform:scale(1);
            transform:scale(1); }
  .bp3-popover-exit-active > .bp3-tooltip{
    -webkit-transform:scale(0.8);
            transform:scale(0.8);
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:100ms;
            transition-duration:100ms;
    -webkit-transition-property:-webkit-transform;
    transition-property:-webkit-transform;
    transition-property:transform;
    transition-property:transform, -webkit-transform;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tooltip .bp3-popover-content{
    padding:10px 12px; }
  .bp3-tooltip.bp3-dark,
  .bp3-dark .bp3-tooltip{
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 2px 4px rgba(16, 22, 26, 0.4), 0 8px 24px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-content,
    .bp3-dark .bp3-tooltip .bp3-popover-content{
      background:#e1e8ed;
      color:#394b59; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow::before,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow::before{
      -webkit-box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4);
              box-shadow:1px 1px 6px rgba(16, 22, 26, 0.4); }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-border,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-border{
      fill:#10161a;
      fill-opacity:0.2; }
    .bp3-tooltip.bp3-dark .bp3-popover-arrow-fill,
    .bp3-dark .bp3-tooltip .bp3-popover-arrow-fill{
      fill:#e1e8ed; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-content{
    background:#137cbd;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-primary .bp3-popover-arrow-fill{
    fill:#137cbd; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-content{
    background:#0f9960;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-success .bp3-popover-arrow-fill{
    fill:#0f9960; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-content{
    background:#d9822b;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-warning .bp3-popover-arrow-fill{
    fill:#d9822b; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-content{
    background:#db3737;
    color:#ffffff; }
  .bp3-tooltip.bp3-intent-danger .bp3-popover-arrow-fill{
    fill:#db3737; }

.bp3-tooltip-indicator{
  border-bottom:dotted 1px;
  cursor:help; }
.bp3-tree .bp3-icon, .bp3-tree .bp3-icon-standard, .bp3-tree .bp3-icon-large{
  color:#5c7080; }
  .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-tree .bp3-icon.bp3-intent-success, .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-tree-node-list{
  list-style:none;
  margin:0;
  padding-left:0; }

.bp3-tree-root{
  background-color:transparent;
  cursor:default;
  padding-left:0;
  position:relative; }

.bp3-tree-node-content-0{
  padding-left:0px; }

.bp3-tree-node-content-1{
  padding-left:23px; }

.bp3-tree-node-content-2{
  padding-left:46px; }

.bp3-tree-node-content-3{
  padding-left:69px; }

.bp3-tree-node-content-4{
  padding-left:92px; }

.bp3-tree-node-content-5{
  padding-left:115px; }

.bp3-tree-node-content-6{
  padding-left:138px; }

.bp3-tree-node-content-7{
  padding-left:161px; }

.bp3-tree-node-content-8{
  padding-left:184px; }

.bp3-tree-node-content-9{
  padding-left:207px; }

.bp3-tree-node-content-10{
  padding-left:230px; }

.bp3-tree-node-content-11{
  padding-left:253px; }

.bp3-tree-node-content-12{
  padding-left:276px; }

.bp3-tree-node-content-13{
  padding-left:299px; }

.bp3-tree-node-content-14{
  padding-left:322px; }

.bp3-tree-node-content-15{
  padding-left:345px; }

.bp3-tree-node-content-16{
  padding-left:368px; }

.bp3-tree-node-content-17{
  padding-left:391px; }

.bp3-tree-node-content-18{
  padding-left:414px; }

.bp3-tree-node-content-19{
  padding-left:437px; }

.bp3-tree-node-content-20{
  padding-left:460px; }

.bp3-tree-node-content{
  -webkit-box-align:center;
      -ms-flex-align:center;
          align-items:center;
  display:-webkit-box;
  display:-ms-flexbox;
  display:flex;
  height:30px;
  padding-right:5px;
  width:100%; }
  .bp3-tree-node-content:hover{
    background-color:rgba(191, 204, 214, 0.4); }

.bp3-tree-node-caret,
.bp3-tree-node-caret-none{
  min-width:30px; }

.bp3-tree-node-caret{
  color:#5c7080;
  cursor:pointer;
  padding:7px;
  -webkit-transform:rotate(0deg);
          transform:rotate(0deg);
  -webkit-transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:-webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9);
  transition:transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9), -webkit-transform 200ms cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-tree-node-caret:hover{
    color:#182026; }
  .bp3-dark .bp3-tree-node-caret{
    color:#a7b6c2; }
    .bp3-dark .bp3-tree-node-caret:hover{
      color:#f5f8fa; }
  .bp3-tree-node-caret.bp3-tree-node-caret-open{
    -webkit-transform:rotate(90deg);
            transform:rotate(90deg); }
  .bp3-tree-node-caret.bp3-icon-standard::before{
    content:""; }

.bp3-tree-node-icon{
  margin-right:7px;
  position:relative; }

.bp3-tree-node-label{
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  word-wrap:normal;
  -webkit-box-flex:1;
      -ms-flex:1 1 auto;
          flex:1 1 auto;
  position:relative;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-label span{
    display:inline; }

.bp3-tree-node-secondary-label{
  padding:0 5px;
  -webkit-user-select:none;
     -moz-user-select:none;
      -ms-user-select:none;
          user-select:none; }
  .bp3-tree-node-secondary-label .bp3-popover-wrapper,
  .bp3-tree-node-secondary-label .bp3-popover-target{
    -webkit-box-align:center;
        -ms-flex-align:center;
            align-items:center;
    display:-webkit-box;
    display:-ms-flexbox;
    display:flex; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-content{
  background-color:inherit;
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-disabled .bp3-tree-node-caret,
.bp3-tree-node.bp3-disabled .bp3-tree-node-icon{
  color:rgba(92, 112, 128, 0.6);
  cursor:not-allowed; }

.bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content,
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-standard, .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-icon-large{
    color:#ffffff; }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret::before{
    color:rgba(255, 255, 255, 0.7); }
  .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content .bp3-tree-node-caret:hover::before{
    color:#ffffff; }

.bp3-dark .bp3-tree-node-content:hover{
  background-color:rgba(92, 112, 128, 0.3); }

.bp3-dark .bp3-tree .bp3-icon, .bp3-dark .bp3-tree .bp3-icon-standard, .bp3-dark .bp3-tree .bp3-icon-large{
  color:#a7b6c2; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-primary, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-primary{
    color:#137cbd; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-success, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-success{
    color:#0f9960; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-warning, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-warning{
    color:#d9822b; }
  .bp3-dark .bp3-tree .bp3-icon.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-standard.bp3-intent-danger, .bp3-dark .bp3-tree .bp3-icon-large.bp3-intent-danger{
    color:#db3737; }

.bp3-dark .bp3-tree-node.bp3-tree-node-selected > .bp3-tree-node-content{
  background-color:#137cbd; }
.bp3-omnibar{
  -webkit-filter:blur(0);
          filter:blur(0);
  opacity:1;
  background-color:#ffffff;
  border-radius:3px;
  -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
          box-shadow:0 0 0 1px rgba(16, 22, 26, 0.1), 0 4px 8px rgba(16, 22, 26, 0.2), 0 18px 46px 6px rgba(16, 22, 26, 0.2);
  left:calc(50% - 250px);
  top:20vh;
  width:500px;
  z-index:21; }
  .bp3-omnibar.bp3-overlay-enter, .bp3-omnibar.bp3-overlay-appear{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2; }
  .bp3-omnibar.bp3-overlay-enter-active, .bp3-omnibar.bp3-overlay-appear-active{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar.bp3-overlay-exit{
    -webkit-filter:blur(0);
            filter:blur(0);
    opacity:1; }
  .bp3-omnibar.bp3-overlay-exit-active{
    -webkit-filter:blur(20px);
            filter:blur(20px);
    opacity:0.2;
    -webkit-transition-delay:0;
            transition-delay:0;
    -webkit-transition-duration:200ms;
            transition-duration:200ms;
    -webkit-transition-property:opacity, -webkit-filter;
    transition-property:opacity, -webkit-filter;
    transition-property:filter, opacity;
    transition-property:filter, opacity, -webkit-filter;
    -webkit-transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9);
            transition-timing-function:cubic-bezier(0.4, 1, 0.75, 0.9); }
  .bp3-omnibar .bp3-input{
    background-color:transparent;
    border-radius:0; }
    .bp3-omnibar .bp3-input, .bp3-omnibar .bp3-input:focus{
      -webkit-box-shadow:none;
              box-shadow:none; }
  .bp3-omnibar .bp3-menu{
    background-color:transparent;
    border-radius:0;
    -webkit-box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
            box-shadow:inset 0 1px 0 rgba(16, 22, 26, 0.15);
    max-height:calc(60vh - 40px);
    overflow:auto; }
    .bp3-omnibar .bp3-menu:empty{
      display:none; }
  .bp3-dark .bp3-omnibar, .bp3-omnibar.bp3-dark{
    background-color:#30404d;
    -webkit-box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4);
            box-shadow:0 0 0 1px rgba(16, 22, 26, 0.2), 0 4px 8px rgba(16, 22, 26, 0.4), 0 18px 46px 6px rgba(16, 22, 26, 0.4); }

.bp3-omnibar-overlay .bp3-overlay-backdrop{
  background-color:rgba(16, 22, 26, 0.2); }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }

.bp3-multi-select{
  min-width:150px; }

.bp3-multi-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto; }

.bp3-select-popover .bp3-popover-content{
  padding:5px; }

.bp3-select-popover .bp3-input-group{
  margin-bottom:0; }

.bp3-select-popover .bp3-menu{
  max-height:300px;
  max-width:400px;
  overflow:auto;
  padding:0; }
  .bp3-select-popover .bp3-menu:not(:first-child){
    padding-top:5px; }
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1pY29uLWJyYW5kMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNmZmYiPgogICAgPHBhdGggZD0iTTEwNSAxMjcuM2g0MHYxMi44aC00MHpNNTEuMSA3N0w3NCA5OS45bC0yMy4zIDIzLjMgMTAuNSAxMC41IDIzLjMtMjMuM0w5NSA5OS45IDg0LjUgODkuNCA2MS42IDY2LjV6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMSBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNGOUE4MjUiPgogICAgPHBhdGggZD0iTTIwLjIgMTEuOGMtMS42IDAtMS43LjUtMS43IDEgMCAuNC4xLjkuMSAxLjMuMS41LjEuOS4xIDEuMyAwIDEuNy0xLjQgMi4zLTMuNSAyLjNoLS45di0xLjloLjVjMS4xIDAgMS40IDAgMS40LS44IDAtLjMgMC0uNi0uMS0xIDAtLjQtLjEtLjgtLjEtMS4yIDAtMS4zIDAtMS44IDEuMy0yLTEuMy0uMi0xLjMtLjctMS4zLTIgMC0uNC4xLS44LjEtMS4yLjEtLjQuMS0uNy4xLTEgMC0uOC0uNC0uNy0xLjQtLjhoLS41VjQuMWguOWMyLjIgMCAzLjUuNyAzLjUgMi4zIDAgLjQtLjEuOS0uMSAxLjMtLjEuNS0uMS45LS4xIDEuMyAwIC41LjIgMSAxLjcgMXYxLjh6TTEuOCAxMC4xYzEuNiAwIDEuNy0uNSAxLjctMSAwLS40LS4xLS45LS4xLTEuMy0uMS0uNS0uMS0uOS0uMS0xLjMgMC0xLjYgMS40LTIuMyAzLjUtMi4zaC45djEuOWgtLjVjLTEgMC0xLjQgMC0xLjQuOCAwIC4zIDAgLjYuMSAxIDAgLjIuMS42LjEgMSAwIDEuMyAwIDEuOC0xLjMgMkM2IDExLjIgNiAxMS43IDYgMTNjMCAuNC0uMS44LS4xIDEuMi0uMS4zLS4xLjctLjEgMSAwIC44LjMuOCAxLjQuOGguNXYxLjloLS45Yy0yLjEgMC0zLjUtLjYtMy41LTIuMyAwLS40LjEtLjkuMS0xLjMuMS0uNS4xLS45LjEtMS4zIDAtLjUtLjItMS0xLjctMXYtMS45eiIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSIxMy44IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY3g9IjExIiBjeT0iOC4yIiByPSIyLjEiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgPGcgY2xhc3M9ImpwLWljb24td2FybjAiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4=);
  --jp-icon-listings-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MC45NzggNTAuOTc4IiBzdHlsZT0iZW5hYmxlLWJhY2tncm91bmQ6bmV3IDAgMCA1MC45NzggNTAuOTc4OyIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSI+Cgk8Zz4KCQk8cGF0aCBzdHlsZT0iZmlsbDojMDEwMDAyOyIgZD0iTTQzLjUyLDcuNDU4QzM4LjcxMSwyLjY0OCwzMi4zMDcsMCwyNS40ODksMEMxOC42NywwLDEyLjI2NiwyLjY0OCw3LjQ1OCw3LjQ1OAoJCQljLTkuOTQzLDkuOTQxLTkuOTQzLDI2LjExOSwwLDM2LjA2MmM0LjgwOSw0LjgwOSwxMS4yMTIsNy40NTYsMTguMDMxLDcuNDU4YzAsMCwwLjAwMSwwLDAuMDAyLDAKCQkJYzYuODE2LDAsMTMuMjIxLTIuNjQ4LDE4LjAyOS03LjQ1OGM0LjgwOS00LjgwOSw3LjQ1Ny0xMS4yMTIsNy40NTctMTguMDNDNTAuOTc3LDE4LjY3LDQ4LjMyOCwxMi4yNjYsNDMuNTIsNy40NTh6CgkJCSBNNDIuMTA2LDQyLjEwNWMtNC40MzIsNC40MzEtMTAuMzMyLDYuODcyLTE2LjYxNSw2Ljg3MmgtMC4wMDJjLTYuMjg1LTAuMDAxLTEyLjE4Ny0yLjQ0MS0xNi42MTctNi44NzIKCQkJYy05LjE2Mi05LjE2My05LjE2Mi0yNC4wNzEsMC0zMy4yMzNDMTMuMzAzLDQuNDQsMTkuMjA0LDIsMjUuNDg5LDJjNi4yODQsMCwxMi4xODYsMi40NCwxNi42MTcsNi44NzIKCQkJYzQuNDMxLDQuNDMxLDYuODcxLDEwLjMzMiw2Ljg3MSwxNi42MTdDNDguOTc3LDMxLjc3Miw0Ni41MzYsMzcuNjc1LDQyLjEwNiw0Mi4xMDV6Ii8+CgkJPHBhdGggc3R5bGU9ImZpbGw6IzAxMDAwMjsiIGQ9Ik0yMy41NzgsMzIuMjE4Yy0wLjAyMy0xLjczNCwwLjE0My0zLjA1OSwwLjQ5Ni0zLjk3MmMwLjM1My0wLjkxMywxLjExLTEuOTk3LDIuMjcyLTMuMjUzCgkJCWMwLjQ2OC0wLjUzNiwwLjkyMy0xLjA2MiwxLjM2Ny0xLjU3NWMwLjYyNi0wLjc1MywxLjEwNC0xLjQ3OCwxLjQzNi0yLjE3NWMwLjMzMS0wLjcwNywwLjQ5NS0xLjU0MSwwLjQ5NS0yLjUKCQkJYzAtMS4wOTYtMC4yNi0yLjA4OC0wLjc3OS0yLjk3OWMtMC41NjUtMC44NzktMS41MDEtMS4zMzYtMi44MDYtMS4zNjljLTEuODAyLDAuMDU3LTIuOTg1LDAuNjY3LTMuNTUsMS44MzIKCQkJYy0wLjMwMSwwLjUzNS0wLjUwMywxLjE0MS0wLjYwNywxLjgxNGMtMC4xMzksMC43MDctMC4yMDcsMS40MzItMC4yMDcsMi4xNzRoLTIuOTM3Yy0wLjA5MS0yLjIwOCwwLjQwNy00LjExNCwxLjQ5My01LjcxOQoJCQljMS4wNjItMS42NCwyLjg1NS0yLjQ4MSw1LjM3OC0yLjUyN2MyLjE2LDAuMDIzLDMuODc0LDAuNjA4LDUuMTQxLDEuNzU4YzEuMjc4LDEuMTYsMS45MjksMi43NjQsMS45NSw0LjgxMQoJCQljMCwxLjE0Mi0wLjEzNywyLjExMS0wLjQxLDIuOTExYy0wLjMwOSwwLjg0NS0wLjczMSwxLjU5My0xLjI2OCwyLjI0M2MtMC40OTIsMC42NS0xLjA2OCwxLjMxOC0xLjczLDIuMDAyCgkJCWMtMC42NSwwLjY5Ny0xLjMxMywxLjQ3OS0xLjk4NywyLjM0NmMtMC4yMzksMC4zNzctMC40MjksMC43NzctMC41NjUsMS4xOTljLTAuMTYsMC45NTktMC4yMTcsMS45NTEtMC4xNzEsMi45NzkKCQkJQzI2LjU4OSwzMi4yMTgsMjMuNTc4LDMyLjIxOCwyMy41NzgsMzIuMjE4eiBNMjMuNTc4LDM4LjIydi0zLjQ4NGgzLjA3NnYzLjQ4NEgyMy41Nzh6Ii8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMEQ0N0ExIj4KICAgIDxwYXRoIGQ9Ik0xMS4xIDYuOVY1LjhINi45YzAtLjUgMC0xLjMuMi0xLjYuNC0uNy44LTEuMSAxLjctMS40IDEuNy0uMyAyLjUtLjMgMy45LS4xIDEgLjEgMS45LjkgMS45IDEuOXY0LjJjMCAuNS0uOSAxLjYtMiAxLjZIOC44Yy0xLjUgMC0yLjQgMS40LTIuNCAyLjh2Mi4ySDQuN0MzLjUgMTUuMSAzIDE0IDMgMTMuMVY5Yy0uMS0xIC42LTIgMS44LTIgMS41LS4xIDYuMy0uMSA2LjMtLjF6Ii8+CiAgICA8cGF0aCBkPSJNMTAuOSAxNS4xdjEuMWg0LjJjMCAuNSAwIDEuMy0uMiAxLjYtLjQuNy0uOCAxLjEtMS43IDEuNC0xLjcuMy0yLjUuMy0zLjkuMS0xLS4xLTEuOS0uOS0xLjktMS45di00LjJjMC0uNS45LTEuNiAyLTEuNmgzLjhjMS41IDAgMi40LTEuNCAyLjQtMi44VjYuNmgxLjdDMTguNSA2LjkgMTkgOCAxOSA4LjlWMTNjMCAxLS43IDIuMS0xLjkgMi4xaC02LjJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIGZpbGw9IiMzMzMzMzMiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uLWFjY2VudDIganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGQ9Ik01LjA1NjY0IDguNzYxNzJDNS4wNTY2NCA4LjU5NzY2IDUuMDMxMjUgOC40NTMxMiA0Ljk4MDQ3IDguMzI4MTJDNC45MzM1OSA4LjE5OTIyIDQuODU1NDcgOC4wODIwMyA0Ljc0NjA5IDcuOTc2NTZDNC42NDA2MiA3Ljg3MTA5IDQuNSA3Ljc3NTM5IDQuMzI0MjIgNy42ODk0NUM0LjE1MjM0IDcuNTk5NjEgMy45NDMzNiA3LjUxMTcyIDMuNjk3MjcgNy40MjU3OEMzLjMwMjczIDcuMjg1MTYgMi45NDMzNiA3LjEzNjcyIDIuNjE5MTQgNi45ODA0N0MyLjI5NDkyIDYuODI0MjIgMi4wMTc1OCA2LjY0MjU4IDEuNzg3MTEgNi40MzU1NUMxLjU2MDU1IDYuMjI4NTIgMS4zODQ3NyA1Ljk4ODI4IDEuMjU5NzcgNS43MTQ4NEMxLjEzNDc3IDUuNDM3NSAxLjA3MjI3IDUuMTA5MzggMS4wNzIyNyA0LjczMDQ3QzEuMDcyMjcgNC4zOTg0NCAxLjEyODkxIDQuMDk1NyAxLjI0MjE5IDMuODIyMjdDMS4zNTU0NyAzLjU0NDkyIDEuNTE1NjIgMy4zMDQ2OSAxLjcyMjY2IDMuMTAxNTZDMS45Mjk2OSAyLjg5ODQ0IDIuMTc5NjkgMi43MzQzNyAyLjQ3MjY2IDIuNjA5MzhDMi43NjU2MiAyLjQ4NDM4IDMuMDkxOCAyLjQwNDMgMy40NTExNyAyLjM2OTE0VjEuMTA5MzhINC4zODg2N1YyLjM4MDg2QzQuNzQwMjMgMi40Mjc3MyA1LjA1NjY0IDIuNTIzNDQgNS4zMzc4OSAyLjY2Nzk3QzUuNjE5MTQgMi44MTI1IDUuODU3NDIgMy4wMDE5NSA2LjA1MjczIDMuMjM2MzNDNi4yNTE5NSAzLjQ2NjggNi40MDQzIDMuNzQwMjMgNi41MDk3NyA0LjA1NjY0QzYuNjE5MTQgNC4zNjkxNCA2LjY3MzgzIDQuNzIwNyA2LjY3MzgzIDUuMTExMzNINS4wNDQ5MkM1LjA0NDkyIDQuNjM4NjcgNC45Mzc1IDQuMjgxMjUgNC43MjI2NiA0LjAzOTA2QzQuNTA3ODEgMy43OTI5NyA0LjIxNjggMy42Njk5MiAzLjg0OTYxIDMuNjY5OTJDMy42NTAzOSAzLjY2OTkyIDMuNDc2NTYgMy42OTcyNyAzLjMyODEyIDMuNzUxOTVDMy4xODM1OSAzLjgwMjczIDMuMDY0NDUgMy44NzY5NSAyLjk3MDcgMy45NzQ2MUMyLjg3Njk1IDQuMDY4MzYgMi44MDY2NCA0LjE3OTY5IDIuNzU5NzcgNC4zMDg1OUMyLjcxNjggNC40Mzc1IDIuNjk1MzEgNC41NzgxMiAyLjY5NTMxIDQuNzMwNDdDMi42OTUzMSA0Ljg4MjgxIDIuNzE2OCA1LjAxOTUzIDIuNzU5NzcgNS4xNDA2MkMyLjgwNjY0IDUuMjU3ODEgMi44ODI4MSA1LjM2NzE5IDIuOTg4MjggNS40Njg3NUMzLjA5NzY2IDUuNTcwMzEgMy4yNDAyMyA1LjY2Nzk3IDMuNDE2MDIgNS43NjE3MkMzLjU5MTggNS44NTE1NiAzLjgxMDU1IDUuOTQzMzYgNC4wNzIyNyA2LjAzNzExQzQuNDY2OCA2LjE4NTU1IDQuODI0MjIgNi4zMzk4NCA1LjE0NDUzIDYuNUM1LjQ2NDg0IDYuNjU2MjUgNS43MzgyOCA2LjgzOTg0IDUuOTY0ODQgNy4wNTA3OEM2LjE5NTMxIDcuMjU3ODEgNi4zNzEwOSA3LjUgNi40OTIxOSA3Ljc3NzM0QzYuNjE3MTkgOC4wNTA3OCA2LjY3OTY5IDguMzc1IDYuNjc5NjkgOC43NUM2LjY3OTY5IDkuMDkzNzUgNi42MjMwNSA5LjQwNDMgNi41MDk3NyA5LjY4MTY0QzYuMzk2NDggOS45NTUwOCA2LjIzNDM4IDEwLjE5MTQgNi4wMjM0NCAxMC4zOTA2QzUuODEyNSAxMC41ODk4IDUuNTU4NTkgMTAuNzUgNS4yNjE3MiAxMC44NzExQzQuOTY0ODQgMTAuOTg4MyA0LjYzMjgxIDExLjA2NDUgNC4yNjU2MiAxMS4wOTk2VjEyLjI0OEgzLjMzMzk4VjExLjA5OTZDMy4wMDE5NSAxMS4wNjg0IDIuNjc5NjkgMTAuOTk2MSAyLjM2NzE5IDEwLjg4MjhDMi4wNTQ2OSAxMC43NjU2IDEuNzc3MzQgMTAuNTk3NyAxLjUzNTE2IDEwLjM3ODlDMS4yOTY4OCAxMC4xNjAyIDEuMTA1NDcgOS44ODQ3NyAwLjk2MDkzOCA5LjU1MjczQzAuODE2NDA2IDkuMjE2OCAwLjc0NDE0MSA4LjgxNDQ1IDAuNzQ0MTQxIDguMzQ1N0gyLjM3ODkxQzIuMzc4OTEgOC42MjY5NSAyLjQxOTkyIDguODYzMjggMi41MDE5NSA5LjA1NDY5QzIuNTgzOTggOS4yNDIxOSAyLjY4OTQ1IDkuMzkyNTggMi44MTgzNiA5LjUwNTg2QzIuOTUxMTcgOS42MTUyMyAzLjEwMTU2IDkuNjkzMzYgMy4yNjk1MyA5Ljc0MDIzQzMuNDM3NSA5Ljc4NzExIDMuNjA5MzggOS44MTA1NSAzLjc4NTE2IDkuODEwNTVDNC4yMDMxMiA5LjgxMDU1IDQuNTE5NTMgOS43MTI4OSA0LjczNDM4IDkuNTE3NThDNC45NDkyMiA5LjMyMjI3IDUuMDU2NjQgOS4wNzAzMSA1LjA1NjY0IDguNzYxNzJaTTEzLjQxOCAxMi4yNzE1SDguMDc0MjJWMTFIMTMuNDE4VjEyLjI3MTVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzLjk1MjY0IDYpIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTUgMTVIM3YyaDEydi0yem0wLThIM3YyaDEyVjd6TTMgMTNoMTh2LTJIM3Yyem0wIDhoMTh2LTJIM3Yyek0zIDN2MmgxOFYzSDN6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4=);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}
.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}
.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}
.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}
.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}
.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}
.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}
.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}
.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}
.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}
.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}
.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}
.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}
.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}
.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}
.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}
.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}
.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}
.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}
.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}
.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}
.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}
.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}
.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}
.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}
.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}
.jp-FileIcon {
  background-image: var(--jp-icon-file);
}
.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}
.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}
.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}
.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}
.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}
.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}
.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}
.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}
.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}
.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}
.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}
.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}
.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}
.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}
.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}
.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}
.jp-ListIcon {
  background-image: var(--jp-icon-list);
}
.jp-ListingsInfoIcon {
  background-image: var(--jp-icon-listings-info);
}
.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}
.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}
.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}
.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}
.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}
.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}
.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}
.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}
.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}
.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}
.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}
.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}
.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}
.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}
.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}
.jp-RunIcon {
  background-image: var(--jp-icon-run);
}
.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}
.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}
.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}
.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}
.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}
.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}
.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}
.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}
.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}
.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}
.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}
.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}
.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}
.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}
.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}
.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}
.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}
/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}
/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}
.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}
.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}
.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}
.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}
.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}
.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}
.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}
.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}
/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}
.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}
.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}
.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}
.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}
.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}
.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}
/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}
.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}
.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}
.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

/* CSS for icons in selected items in the settings editor */
#setting-editor .jp-PluginList .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
#setting-editor
  .jp-PluginList
  .jp-mod-selected
  .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* CSS for icons in selected tabs in the sidebar tab manager */
#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable[fill] {
  fill: #fff;
}

#tab-manager .lm-TabBar-tab.jp-mod-active .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable[fill] {
  fill: var(--jp-brand-color1);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-active
  .jp-icon-hover
  :hover
  .jp-icon-selectable-inverse[fill] {
  fill: #fff;
}

/**
 * TODO: come up with non css-hack solution for showing the busy icon on top
 *  of the close icon
 * CSS for complex behavior of close icon of tabs in the sidebar tab manager
 */
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
#tab-manager
  .lm-TabBar-tab.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

#tab-manager
  .lm-TabBar-tab.jp-mod-dirty.jp-mod-active
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: #fff;
}

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}
/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) svg {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-border-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0px;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-warn-color0);
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

/* Override Blueprint's _reset.scss styles */
html {
  box-sizing: unset;
}

*,
*::before,
*::after {
  box-sizing: unset;
}

body {
  color: unset;
  font-family: var(--jp-ui-font-family);
}

p {
  margin-top: unset;
  margin-bottom: unset;
}

small {
  font-size: unset;
}

strong {
  font-weight: unset;
}

/* Override Blueprint's _typography.scss styles */
a {
  text-decoration: unset;
  color: unset;
}
a:hover {
  text-decoration: unset;
  color: unset;
}

/* Override Blueprint's _accessibility.scss styles */
:focus {
  outline: unset;
  outline-offset: unset;
  -moz-outline-radius: unset;
}

/* Styles for ui-components */
.jp-Button {
  border-radius: var(--jp-border-radius);
  padding: 0px 12px;
  font-size: var(--jp-ui-font-size1);
}

/* Use our own theme for hover styles */
button.jp-Button.bp3-button.bp3-minimal:hover {
  background-color: var(--jp-layout-color2);
}
.jp-Button.minimal {
  color: unset !important;
}

.jp-Button.jp-ToolbarButtonComponent {
  text-transform: none;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color3);
}

.jp-BPIcon {
  display: inline-block;
  vertical-align: middle;
  margin: auto;
}

/* Stop blueprint futzing with our icon fills */
.bp3-icon.jp-BPIcon > svg:not([fill]) {
  fill: var(--jp-inverse-layout-color3);
}

.jp-InputGroupAction {
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

/* Use our own theme for hover and option styles */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}
select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-top: 1px solid var(--jp-border-color2);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-Collapse-header {
  padding: 1px 12px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size2);
}

.jp-Collapse-header:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Collapse-contents {
  padding: 0px 12px 0px 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0px;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0px 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px 5px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0px;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty:after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0px 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0px;
  left: 0px;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px;
  padding-bottom: 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0px;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0px 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

.jp-HoverBox.jp-mod-outofview {
  display: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;

  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;

  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #aa00ff;

  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;

  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;

  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;

  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;

  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;

  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;

  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;

  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;

  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;

  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ffff00;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;

  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;

  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;

  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;

  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;

  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eeeeee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent:before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent:after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0px 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FileDialog-Checkbox {
  margin-top: 35px;
  display: flex;
  flex-direction: row;
  align-items: end;
  width: 100%;
}

.jp-FileDialog-Checkbox > label {
  flex: 1 1 auto;
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  height: 28px;
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  background-color: var(--jp-layout-color1);
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0px 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  height: 32px;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 1;
  overflow-x: auto;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px;
  margin: 0px;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0px 6px;
  margin: 0px;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent span {
  padding: 0px;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/


/* <DEPRECATED> */ body.p-mod-override-cursor *, /* </DEPRECATED> */
body.lm-mod-override-cursor * {
  cursor: inherit !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0px;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/* BASICS */

.CodeMirror {
  /* Set height, width, borders, and global font properties here */
  font-family: monospace;
  height: 300px;
  color: black;
  direction: ltr;
}

/* PADDING */

.CodeMirror-lines {
  padding: 4px 0; /* Vertical padding around content */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  padding: 0 4px; /* Horizontal padding of content */
}

.CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  background-color: white; /* The little square between H and V scrollbars */
}

/* GUTTER */

.CodeMirror-gutters {
  border-right: 1px solid #ddd;
  background-color: #f7f7f7;
  white-space: nowrap;
}
.CodeMirror-linenumbers {}
.CodeMirror-linenumber {
  padding: 0 3px 0 5px;
  min-width: 20px;
  text-align: right;
  color: #999;
  white-space: nowrap;
}

.CodeMirror-guttermarker { color: black; }
.CodeMirror-guttermarker-subtle { color: #999; }

/* CURSOR */

.CodeMirror-cursor {
  border-left: 1px solid black;
  border-right: none;
  width: 0;
}
/* Shown when moving in bi-directional text */
.CodeMirror div.CodeMirror-secondarycursor {
  border-left: 1px solid silver;
}
.cm-fat-cursor .CodeMirror-cursor {
  width: auto;
  border: 0 !important;
  background: #7e7;
}
.cm-fat-cursor div.CodeMirror-cursors {
  z-index: 1;
}
.cm-fat-cursor-mark {
  background-color: rgba(20, 255, 20, 0.5);
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
}
.cm-animate-fat-cursor {
  width: auto;
  border: 0;
  -webkit-animation: blink 1.06s steps(1) infinite;
  -moz-animation: blink 1.06s steps(1) infinite;
  animation: blink 1.06s steps(1) infinite;
  background-color: #7e7;
}
@-moz-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@-webkit-keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}
@keyframes blink {
  0% {}
  50% { background-color: transparent; }
  100% {}
}

/* Can style cursor different in overwrite (non-insert) mode */
.CodeMirror-overwrite .CodeMirror-cursor {}

.cm-tab { display: inline-block; text-decoration: inherit; }

.CodeMirror-rulers {
  position: absolute;
  left: 0; right: 0; top: -50px; bottom: 0;
  overflow: hidden;
}
.CodeMirror-ruler {
  border-left: 1px solid #ccc;
  top: 0; bottom: 0;
  position: absolute;
}

/* DEFAULT THEME */

.cm-s-default .cm-header {color: blue;}
.cm-s-default .cm-quote {color: #090;}
.cm-negative {color: #d44;}
.cm-positive {color: #292;}
.cm-header, .cm-strong {font-weight: bold;}
.cm-em {font-style: italic;}
.cm-link {text-decoration: underline;}
.cm-strikethrough {text-decoration: line-through;}

.cm-s-default .cm-keyword {color: #708;}
.cm-s-default .cm-atom {color: #219;}
.cm-s-default .cm-number {color: #164;}
.cm-s-default .cm-def {color: #00f;}
.cm-s-default .cm-variable,
.cm-s-default .cm-punctuation,
.cm-s-default .cm-property,
.cm-s-default .cm-operator {}
.cm-s-default .cm-variable-2 {color: #05a;}
.cm-s-default .cm-variable-3, .cm-s-default .cm-type {color: #085;}
.cm-s-default .cm-comment {color: #a50;}
.cm-s-default .cm-string {color: #a11;}
.cm-s-default .cm-string-2 {color: #f50;}
.cm-s-default .cm-meta {color: #555;}
.cm-s-default .cm-qualifier {color: #555;}
.cm-s-default .cm-builtin {color: #30a;}
.cm-s-default .cm-bracket {color: #997;}
.cm-s-default .cm-tag {color: #170;}
.cm-s-default .cm-attribute {color: #00c;}
.cm-s-default .cm-hr {color: #999;}
.cm-s-default .cm-link {color: #00c;}

.cm-s-default .cm-error {color: #f00;}
.cm-invalidchar {color: #f00;}

.CodeMirror-composing { border-bottom: 2px solid; }

/* Default styles for common addons */

div.CodeMirror span.CodeMirror-matchingbracket {color: #0b0;}
div.CodeMirror span.CodeMirror-nonmatchingbracket {color: #a22;}
.CodeMirror-matchingtag { background: rgba(255, 150, 0, .3); }
.CodeMirror-activeline-background {background: #e8f2ff;}

/* STOP */

/* The rest of this file contains styles related to the mechanics of
   the editor. You probably shouldn't touch them. */

.CodeMirror {
  position: relative;
  overflow: hidden;
  background: white;
}

.CodeMirror-scroll {
  overflow: scroll !important; /* Things will break if this is overridden */
  /* 50px is the magic margin used to hide the element's real scrollbars */
  /* See overflow: hidden in .CodeMirror */
  margin-bottom: -50px; margin-right: -50px;
  padding-bottom: 50px;
  height: 100%;
  outline: none; /* Prevent dragging from highlighting the element */
  position: relative;
}
.CodeMirror-sizer {
  position: relative;
  border-right: 50px solid transparent;
}

/* The fake, visible scrollbars. Used to force redraw during scrolling
   before actual scrolling happens, thus preventing shaking and
   flickering artifacts. */
.CodeMirror-vscrollbar, .CodeMirror-hscrollbar, .CodeMirror-scrollbar-filler, .CodeMirror-gutter-filler {
  position: absolute;
  z-index: 6;
  display: none;
  outline: none;
}
.CodeMirror-vscrollbar {
  right: 0; top: 0;
  overflow-x: hidden;
  overflow-y: scroll;
}
.CodeMirror-hscrollbar {
  bottom: 0; left: 0;
  overflow-y: hidden;
  overflow-x: scroll;
}
.CodeMirror-scrollbar-filler {
  right: 0; bottom: 0;
}
.CodeMirror-gutter-filler {
  left: 0; bottom: 0;
}

.CodeMirror-gutters {
  position: absolute; left: 0; top: 0;
  min-height: 100%;
  z-index: 3;
}
.CodeMirror-gutter {
  white-space: normal;
  height: 100%;
  display: inline-block;
  vertical-align: top;
  margin-bottom: -50px;
}
.CodeMirror-gutter-wrapper {
  position: absolute;
  z-index: 4;
  background: none !important;
  border: none !important;
}
.CodeMirror-gutter-background {
  position: absolute;
  top: 0; bottom: 0;
  z-index: 4;
}
.CodeMirror-gutter-elt {
  position: absolute;
  cursor: default;
  z-index: 4;
}
.CodeMirror-gutter-wrapper ::selection { background-color: transparent }
.CodeMirror-gutter-wrapper ::-moz-selection { background-color: transparent }

.CodeMirror-lines {
  cursor: text;
  min-height: 1px; /* prevents collapsing before first draw */
}
.CodeMirror pre.CodeMirror-line,
.CodeMirror pre.CodeMirror-line-like {
  /* Reset some styles that the rest of the page might have set */
  -moz-border-radius: 0; -webkit-border-radius: 0; border-radius: 0;
  border-width: 0;
  background: transparent;
  font-family: inherit;
  font-size: inherit;
  margin: 0;
  white-space: pre;
  word-wrap: normal;
  line-height: inherit;
  color: inherit;
  z-index: 2;
  position: relative;
  overflow: visible;
  -webkit-tap-highlight-color: transparent;
  -webkit-font-variant-ligatures: contextual;
  font-variant-ligatures: contextual;
}
.CodeMirror-wrap pre.CodeMirror-line,
.CodeMirror-wrap pre.CodeMirror-line-like {
  word-wrap: break-word;
  white-space: pre-wrap;
  word-break: normal;
}

.CodeMirror-linebackground {
  position: absolute;
  left: 0; right: 0; top: 0; bottom: 0;
  z-index: 0;
}

.CodeMirror-linewidget {
  position: relative;
  z-index: 2;
  padding: 0.1px; /* Force widget margins to stay inside of the container */
}

.CodeMirror-widget {}

.CodeMirror-rtl pre { direction: rtl; }

.CodeMirror-code {
  outline: none;
}

/* Force content-box sizing for the elements where we expect it */
.CodeMirror-scroll,
.CodeMirror-sizer,
.CodeMirror-gutter,
.CodeMirror-gutters,
.CodeMirror-linenumber {
  -moz-box-sizing: content-box;
  box-sizing: content-box;
}

.CodeMirror-measure {
  position: absolute;
  width: 100%;
  height: 0;
  overflow: hidden;
  visibility: hidden;
}

.CodeMirror-cursor {
  position: absolute;
  pointer-events: none;
}
.CodeMirror-measure pre { position: static; }

div.CodeMirror-cursors {
  visibility: hidden;
  position: relative;
  z-index: 3;
}
div.CodeMirror-dragcursors {
  visibility: visible;
}

.CodeMirror-focused div.CodeMirror-cursors {
  visibility: visible;
}

.CodeMirror-selected { background: #d9d9d9; }
.CodeMirror-focused .CodeMirror-selected { background: #d7d4f0; }
.CodeMirror-crosshair { cursor: crosshair; }
.CodeMirror-line::selection, .CodeMirror-line > span::selection, .CodeMirror-line > span > span::selection { background: #d7d4f0; }
.CodeMirror-line::-moz-selection, .CodeMirror-line > span::-moz-selection, .CodeMirror-line > span > span::-moz-selection { background: #d7d4f0; }

.cm-searching {
  background-color: #ffa;
  background-color: rgba(255, 255, 0, .4);
}

/* Used to force a border model for a node */
.cm-force-border { padding-right: .1px; }

@media print {
  /* Hide the cursor when printing */
  .CodeMirror div.CodeMirror-cursors {
    visibility: hidden;
  }
}

/* See issue #2901 */
.cm-tab-wrap-hack:after { content: ''; }

/* Help users use markselection to safely style text background */
span.CodeMirror-selectedtext { background: none; }

.CodeMirror-dialog {
  position: absolute;
  left: 0; right: 0;
  background: inherit;
  z-index: 15;
  padding: .1em .8em;
  overflow: hidden;
  color: inherit;
}

.CodeMirror-dialog-top {
  border-bottom: 1px solid #eee;
  top: 0;
}

.CodeMirror-dialog-bottom {
  border-top: 1px solid #eee;
  bottom: 0;
}

.CodeMirror-dialog input {
  border: none;
  outline: none;
  background: transparent;
  width: 20em;
  color: inherit;
  font-family: monospace;
}

.CodeMirror-dialog button {
  font-size: 70%;
}

.CodeMirror-foldmarker {
  color: blue;
  text-shadow: #b9f 1px 1px 2px, #b9f -1px -1px 2px, #b9f 1px -1px 2px, #b9f -1px 1px 2px;
  font-family: arial;
  line-height: .3;
  cursor: pointer;
}
.CodeMirror-foldgutter {
  width: .7em;
}
.CodeMirror-foldgutter-open,
.CodeMirror-foldgutter-folded {
  cursor: pointer;
}
.CodeMirror-foldgutter-open:after {
  content: "\25BE";
}
.CodeMirror-foldgutter-folded:after {
  content: "\25B8";
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.CodeMirror {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;
  /* Changed to auto to autogrow */
}

.CodeMirror pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* This causes https://github.com/jupyter/jupyterlab/issues/522 */
/* May not cause it not because we changed it! */
.CodeMirror-lines {
  padding: var(--jp-code-padding) 0;
}

.CodeMirror-linenumber {
  padding: 0 8px;
}

.jp-CodeMirrorEditor {
  cursor: text;
}

.jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .CodeMirror-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.CodeMirror.jp-mod-readOnly .CodeMirror-cursor {
  display: none;
}

.CodeMirror-gutters {
  border-right: 1px solid var(--jp-border-color2);
  background-color: var(--jp-layout-color0);
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.CodeMirror-selectedtext.cm-searching {
  background-color: var(--jp-search-selected-match-background-color) !important;
  color: var(--jp-search-selected-match-color) !important;
}

.cm-searching {
  background-color: var(
    --jp-search-unselected-match-background-color
  ) !important;
  color: var(--jp-search-unselected-match-color) !important;
}

.CodeMirror-focused .CodeMirror-selected {
  background-color: var(--jp-editor-selected-focused-background);
}

.CodeMirror-selected {
  background-color: var(--jp-editor-selected-background);
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/**
 * Here is our jupyter theme for CodeMirror syntax highlighting
 * This is used in our marked.js syntax highlighting and CodeMirror itself
 * The string "jupyter" is set in ../codemirror/widget.DEFAULT_CODEMIRROR_THEME
 * This came from the classic notebook, which came form highlight.js/GitHub
 */

/**
 * CodeMirror themes are handling the background/color in this way. This works
 * fine for CodeMirror editors outside the notebook, but the notebook styles
 * these things differently.
 */
.CodeMirror.cm-s-jupyter {
  background: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

/* In the notebook, we want this styling to be handled by its container */
.jp-CodeConsole .CodeMirror.cm-s-jupyter,
.jp-Notebook .CodeMirror.cm-s-jupyter {
  background: transparent;
}

.cm-s-jupyter .CodeMirror-cursor {
  border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);
}
.cm-s-jupyter span.cm-keyword {
  color: var(--jp-mirror-editor-keyword-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-atom {
  color: var(--jp-mirror-editor-atom-color);
}
.cm-s-jupyter span.cm-number {
  color: var(--jp-mirror-editor-number-color);
}
.cm-s-jupyter span.cm-def {
  color: var(--jp-mirror-editor-def-color);
}
.cm-s-jupyter span.cm-variable {
  color: var(--jp-mirror-editor-variable-color);
}
.cm-s-jupyter span.cm-variable-2 {
  color: var(--jp-mirror-editor-variable-2-color);
}
.cm-s-jupyter span.cm-variable-3 {
  color: var(--jp-mirror-editor-variable-3-color);
}
.cm-s-jupyter span.cm-punctuation {
  color: var(--jp-mirror-editor-punctuation-color);
}
.cm-s-jupyter span.cm-property {
  color: var(--jp-mirror-editor-property-color);
}
.cm-s-jupyter span.cm-operator {
  color: var(--jp-mirror-editor-operator-color);
  font-weight: bold;
}
.cm-s-jupyter span.cm-comment {
  color: var(--jp-mirror-editor-comment-color);
  font-style: italic;
}
.cm-s-jupyter span.cm-string {
  color: var(--jp-mirror-editor-string-color);
}
.cm-s-jupyter span.cm-string-2 {
  color: var(--jp-mirror-editor-string-2-color);
}
.cm-s-jupyter span.cm-meta {
  color: var(--jp-mirror-editor-meta-color);
}
.cm-s-jupyter span.cm-qualifier {
  color: var(--jp-mirror-editor-qualifier-color);
}
.cm-s-jupyter span.cm-builtin {
  color: var(--jp-mirror-editor-builtin-color);
}
.cm-s-jupyter span.cm-bracket {
  color: var(--jp-mirror-editor-bracket-color);
}
.cm-s-jupyter span.cm-tag {
  color: var(--jp-mirror-editor-tag-color);
}
.cm-s-jupyter span.cm-attribute {
  color: var(--jp-mirror-editor-attribute-color);
}
.cm-s-jupyter span.cm-header {
  color: var(--jp-mirror-editor-header-color);
}
.cm-s-jupyter span.cm-quote {
  color: var(--jp-mirror-editor-quote-color);
}
.cm-s-jupyter span.cm-link {
  color: var(--jp-mirror-editor-link-color);
}
.cm-s-jupyter span.cm-error {
  color: var(--jp-mirror-editor-error-color);
}
.cm-s-jupyter span.cm-hr {
  color: #999;
}

.cm-s-jupyter span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}

.cm-s-jupyter .CodeMirror-activeline-background,
.cm-s-jupyter .CodeMirror-gutter {
  background-color: var(--jp-layout-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .remote-caret {
  position: relative;
  border-left: 2px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .remote-caret > div {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -2px;
  font-size: 0.95em;
  background-color: rgb(250, 129, 0);
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 3;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .remote-caret.hide-name > div {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .remote-caret:hover > div {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0px;
  padding: 0px;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}
.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}
.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}
.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}
.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}
.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}
.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}
.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}
.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}
.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}
.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}
.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}
.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}
.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}
.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}
.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}
.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}
.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0em;
}

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: 12px;
  table-layout: fixed;
  margin-left: auto;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon table {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0px;
}

.jp-RenderedHTMLCommon p {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}
/* ...or leave it untouched if they don't */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-dark-background {
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}
.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}
.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}
.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}
.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}
.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}
.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}
.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}
.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: 0.8em;
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  border-bottom: none;
  height: auto;
  margin: var(--jp-toolbar-header-margin);
  box-shadow: none;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0px 2px;
  padding: 0px 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0px;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar.jp-Toolbar {
  padding: 0px;
  margin: 8px 12px 0px 12px;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  justify-content: flex-start;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0px;
  padding-right: 2px;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
  width: 40px;
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent {
  width: 72px;
  background: var(--jp-brand-color1);
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent:focus-visible {
  background-color: var(--jp-brand-color0);
}

.jp-FileBrowser-toolbar.jp-Toolbar
  .jp-Toolbar-item:first-child
  .jp-ToolbarButtonComponent
  .jp-icon3 {
  fill: white;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileBrowser-filterBox {
  padding: 0px;
  flex: 0 0 auto;
  margin: 8px 12px 0px 12px;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing:focus-visible {
  border: 1px solid var(--jp-brand-color1);
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px 12px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon:before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon:before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0px;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-DirListing-deadSpace {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
}

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: flex;
  flex-direction: row;
}

body[data-format='mobile'] .jp-OutputArea-child {
  flex-direction: column;
}

.jp-OutputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

body[data-format='mobile'] .jp-OutputPrompt {
  flex: 0 0 auto;
  text-align: left;
}

.jp-OutputArea-output {
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea-child .jp-OutputArea-output {
  flex-grow: 1;
  flex-shrink: 1;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  margin-left: var(--jp-notebook-padding);
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `p-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated:before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0px;
  padding: 0px;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0px;
  flex: 1 1 auto;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-OutputArea-stdin {
  line-height: var(--jp-code-line-height);
  padding-top: var(--jp-code-padding);
  display: flex;
}

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;
  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0px;
  bottom: 0px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0px;
  width: 100%;
  padding: 0px;
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: flex;
  flex-direction: row;
  overflow: hidden;
}

body[data-format='mobile'] .jp-InputArea {
  flex-direction: column;
}

.jp-InputArea-editor {
  flex: 1 1 auto;
  overflow: hidden;
}

.jp-InputArea-editor {
  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0px;
  background: var(--jp-cell-editor-background);
}

body[data-format='mobile'] .jp-InputArea-editor {
  margin-left: var(--jp-notebook-padding);
}

.jp-InputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);
  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

body[data-format='mobile'] .jp-InputPrompt {
  flex: 0 0 auto;
  text-align: left;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: flex;
  flex-direction: row;
  flex: 1 1 auto;
}

.jp-Placeholder-prompt {
  box-sizing: border-box;
}

.jp-Placeholder-content {
  flex: 1 1 auto;
  border: none;
  background: transparent;
  height: 20px;
  box-sizing: border-box;
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0px;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0px;
  margin: 0px;
  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 200px;
  box-shadow: inset 0 0 6px 2px rgba(0, 0, 0, 0.3);
  margin-left: var(--jp-private-cell-scrolling-output-offset);
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  flex: 0 0
    calc(
      var(--jp-cell-prompt-width) -
        var(--jp-private-cell-scrolling-output-offset)
    );
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  flex: 1 1 auto;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

.jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
}

.jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-collapseHeadingButton {
  display: none;
}

.jp-MarkdownCell:hover .jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: 2px;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook-render * {
  contain: none !important;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
  float: left;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt:before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0px;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0px rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-NotebookTools-tool {
  padding: 0px 12px 0 12px;
}

.jp-ActiveCellTool {
  padding: 12px;
  background-color: var(--jp-layout-color1);
  border-top: none !important;
}

.jp-ActiveCellTool .jp-InputArea-prompt {
  flex: 0 0 auto;
  padding-left: 0px;
}

.jp-ActiveCellTool .jp-InputArea-editor {
  flex: 1 1 auto;
  background: var(--jp-cell-editor-background);
  border-color: var(--jp-cell-editor-border-color);
}

.jp-ActiveCellTool .jp-InputArea-editor .CodeMirror {
  background: transparent;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0px 12px 0px;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body div {
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>

    <style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0px 2px 1px -1px var(--jp-shadow-umbra-color),
    0px 1px 1px 0px var(--jp-shadow-penumbra-color),
    0px 1px 3px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0px 3px 1px -2px var(--jp-shadow-umbra-color),
    0px 2px 2px 0px var(--jp-shadow-penumbra-color),
    0px 1px 5px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0px 2px 4px -1px var(--jp-shadow-umbra-color),
    0px 4px 5px 0px var(--jp-shadow-penumbra-color),
    0px 1px 10px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0px 3px 5px -1px var(--jp-shadow-umbra-color),
    0px 6px 10px 0px var(--jp-shadow-penumbra-color),
    0px 1px 18px 0px var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0px 5px 5px -3px var(--jp-shadow-umbra-color),
    0px 8px 10px 1px var(--jp-shadow-penumbra-color),
    0px 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0px 7px 8px -4px var(--jp-shadow-umbra-color),
    0px 12px 17px 2px var(--jp-shadow-penumbra-color),
    0px 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0px 8px 10px -5px var(--jp-shadow-umbra-color),
    0px 16px 24px 2px var(--jp-shadow-penumbra-color),
    0px 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0px 10px 13px -6px var(--jp-shadow-umbra-color),
    0px 20px 31px 3px var(--jp-shadow-penumbra-color),
    0px 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0px 11px 15px -7px var(--jp-shadow-umbra-color),
    0px 24px 38px 3px var(--jp-shadow-penumbra-color),
    0px 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;

  --jp-ui-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica,
    Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;

  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);

  --jp-content-link-color: var(--md-blue-700);

  --jp-content-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: Menlo, Consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);

  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;

  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;

  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);

  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-border-color1);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: #05a;
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #aa22ff;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #aa22ff;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);
}
</style>

<style type="text/css">
/* Misc */
a.anchor-link {
  display: none;
}

.highlight  {
  margin: 0.4em;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

/* Using table instead of flexbox so that we can use break-inside property */
/* CSS rules under this comment should not be required anymore after we move to the JupyterLab 4.0 CSS */


.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  min-width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-OutputArea-child {
  display: table;
  width: 100%;
}

.jp-OutputPrompt {
  display: table-cell;
  vertical-align: top;
  min-width: var(--jp-cell-prompt-width);
}

body[data-format='mobile'] .jp-OutputPrompt {
  display: table-row;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
}

body[data-format='mobile'] .jp-OutputArea-child .jp-OutputArea-output {
  display: table-row;
}

.jp-OutputArea-output.jp-OutputArea-executeResult {
  width: 100%;
}

@media print {
  .jp-Collapser {
    display: none;
  }

  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }

  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}
</style>

<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: { 
                    automatic: true 
                    }
                }
            });
        
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">

<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 align=center>Exploratory Data Analysis on very famous Sample Superstore Dataset</h2>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Link to data source: <a href="https://www.kaggle.com/aungpyaeap/supermarket-sales">https://www.kaggle.com/aungpyaeap/supermarket-sales</a></p>
<p><strong>Context</strong></p>
<p>The growth of supermarkets in most populated cities are increasing and market competitions are also high. The dataset is one of the historical sales of supermarket company which has recorded in 3 different branches for 3 months data.</p>
<p><strong>Data Dictionary</strong></p>
<ol>
<li><p><strong><em>Invoice id:</em></strong> Computer generated sales slip invoice identification number</p>
</li>
<li><p><strong><em>Branch:</em></strong> Branch of supercenter (3 branches are available identified by A, B and C).</p>
</li>
<li><p><strong><em>City:</em></strong> Location of supercenters</p>
</li>
<li><p><strong><em>Customer type:</em></strong> Type of customers, recorded by Members for customers using member card and Normal for without member card.</p>
</li>
<li><p><strong><em>Gender:</em></strong> Gender type of customer</p>
</li>
<li><p><strong><em>Product line:</em></strong> General item categorization groups - Electronic accessories, Fashion accessories, Food and beverages, Health and beauty, Home and lifestyle, Sports and travel</p>
</li>
<li><p><strong><em>Unit price:</em></strong> Price of each product in USD</p>
</li>
<li><p><strong><em>Quantity:</em></strong> Number of products purchased by customer</p>
</li>
<li><p><strong><em>Tax:</em></strong> 5% tax fee for customer buying</p>
</li>
<li><p><strong><em>Total:</em></strong> Total price including tax</p>
</li>
<li><p><strong><em>Date:</em></strong> Date of purchase (Record available from January 2019 to March 2019)</p>
</li>
<li><p><strong><em>Time:</em></strong> Purchase time (10am to 9pm)</p>
</li>
<li><p><strong><em>Payment:</em></strong> Payment used by customer for purchase (3 methods are available – Cash, Credit card and Ewallet)</p>
</li>
<li><p><strong><em>COGS:</em></strong> Cost of goods sold</p>
</li>
<li><p><strong><em>Gross margin percentage:</em></strong> Gross margin percentage</p>
</li>
<li><p><strong><em>Gross income:</em></strong> Gross income</p>
</li>
<li><p><strong><em>Rating:</em></strong> Customer stratification rating on their overall shopping experience (On a scale of 1 to 10)</p>
</li>
</ol>

</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="OK,-lets-start-with-Boilerplate-stuff,-import-the-libraries">OK, lets start with Boilerplate stuff, import the libraries<a class="anchor-link" href="#OK,-lets-start-with-Boilerplate-stuff,-import-the-libraries">&#182;</a></h4>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[1]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="c1">#import the warnings.</span>
<span class="kn">import</span> <span class="nn">warnings</span>
<span class="n">warnings</span><span class="o">.</span><span class="n">filterwarnings</span><span class="p">(</span><span class="s1">&#39;ignore&#39;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Lets-explore-the-data-now,-shall-we?">Lets explore the data now, shall we?<a class="anchor-link" href="#Lets-explore-the-data-now,-shall-we?">&#182;</a></h3>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Load-the-Dataset">Load the Dataset<a class="anchor-link" href="#Load-the-Dataset">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[2]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;supermarket_sales.csv&quot;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Take-a-glimpse-of-the-dataset">Take a glimpse of the dataset<a class="anchor-link" href="#Take-a-glimpse-of-the-dataset">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[3]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[3]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Invoice ID</th>
      <th>Branch</th>
      <th>City</th>
      <th>Customer type</th>
      <th>Gender</th>
      <th>Product line</th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>Date</th>
      <th>Time</th>
      <th>Payment</th>
      <th>cogs</th>
      <th>gross margin percentage</th>
      <th>gross income</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>750-67-8428</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Female</td>
      <td>Health and beauty</td>
      <td>74.69</td>
      <td>7.0</td>
      <td>26.1415</td>
      <td>548.9715</td>
      <td>1/5/19</td>
      <td>13:08</td>
      <td>Ewallet</td>
      <td>522.83</td>
      <td>4.761905</td>
      <td>26.1415</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>226-31-3081</td>
      <td>C</td>
      <td>Naypyitaw</td>
      <td>Normal</td>
      <td>Female</td>
      <td>Electronic accessories</td>
      <td>15.28</td>
      <td>5.0</td>
      <td>3.8200</td>
      <td>80.2200</td>
      <td>3/8/19</td>
      <td>10:29</td>
      <td>Cash</td>
      <td>76.40</td>
      <td>4.761905</td>
      <td>3.8200</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>631-41-3108</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Home and lifestyle</td>
      <td>46.33</td>
      <td>7.0</td>
      <td>16.2155</td>
      <td>340.5255</td>
      <td>3/3/19</td>
      <td>13:23</td>
      <td>Credit card</td>
      <td>324.31</td>
      <td>4.761905</td>
      <td>16.2155</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>123-19-1176</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Male</td>
      <td>Health and beauty</td>
      <td>58.22</td>
      <td>8.0</td>
      <td>23.2880</td>
      <td>489.0480</td>
      <td>1/27/19</td>
      <td>20:33</td>
      <td>Ewallet</td>
      <td>465.76</td>
      <td>4.761905</td>
      <td>23.2880</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>373-73-7910</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Sports and travel</td>
      <td>86.31</td>
      <td>7.0</td>
      <td>30.2085</td>
      <td>634.3785</td>
      <td>2/8/19</td>
      <td>10:37</td>
      <td>Ewallet</td>
      <td>604.17</td>
      <td>4.761905</td>
      <td>30.2085</td>
      <td>5.3</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[4]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span> <span class="n">df</span><span class="o">.</span><span class="n">columns</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[4]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>Index([&#39;Invoice ID&#39;, &#39;Branch&#39;, &#39;City&#39;, &#39;Customer type&#39;, &#39;Gender&#39;,
       &#39;Product line&#39;, &#39;Unit price&#39;, &#39;Quantity&#39;, &#39;Tax 5%&#39;, &#39;Total&#39;, &#39;Date&#39;,
       &#39;Time&#39;, &#39;Payment&#39;, &#39;cogs&#39;, &#39;gross margin percentage&#39;, &#39;gross income&#39;,
       &#39;Rating&#39;],
      dtype=&#39;object&#39;)</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Check-Datatypes-of-columns">Check Datatypes of columns<a class="anchor-link" href="#Check-Datatypes-of-columns">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[5]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">dtypes</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[5]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>Invoice ID                  object
Branch                      object
City                        object
Customer type               object
Gender                      object
Product line                object
Unit price                 float64
Quantity                   float64
Tax 5%                     float64
Total                      float64
Date                        object
Time                        object
Payment                     object
cogs                       float64
gross margin percentage    float64
gross income               float64
Rating                     float64
dtype: object</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Date-column-appears-as-object,-check-that-column's-format">Date column appears as object, check that column's format<a class="anchor-link" href="#Date-column-appears-as-object,-check-that-column's-format">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[6]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">Date</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[6]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>0        1/5/19
1        3/8/19
2        3/3/19
3       1/27/19
4        2/8/19
         ...   
998     2/22/19
999     2/18/19
1000    2/18/19
1001    3/10/19
1002    1/26/19
Name: Date, Length: 1003, dtype: object</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Change-Date-Column-to-DateTime-format">Change Date Column to DateTime format<a class="anchor-link" href="#Change-Date-Column-to-DateTime-format">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[7]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">Date</span><span class="o">=</span><span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Date</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Verify-the-changed-format">Verify the changed format<a class="anchor-link" href="#Verify-the-changed-format">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[8]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">Date</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[8]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>0      2019-01-05
1      2019-03-08
2      2019-03-03
3      2019-01-27
4      2019-02-08
          ...    
998    2019-02-22
999    2019-02-18
1000   2019-02-18
1001   2019-03-10
1002   2019-01-26
Name: Date, Length: 1003, dtype: datetime64[ns]</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="COOL,-date-format-is-better-now...">COOL, date format is better now...<a class="anchor-link" href="#COOL,-date-format-is-better-now...">&#182;</a></h4>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="OK,-lets-see-the-dataset-again.">OK, lets see the dataset again.<a class="anchor-link" href="#OK,-lets-see-the-dataset-again.">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[9]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[9]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Invoice ID</th>
      <th>Branch</th>
      <th>City</th>
      <th>Customer type</th>
      <th>Gender</th>
      <th>Product line</th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>Date</th>
      <th>Time</th>
      <th>Payment</th>
      <th>cogs</th>
      <th>gross margin percentage</th>
      <th>gross income</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>750-67-8428</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Female</td>
      <td>Health and beauty</td>
      <td>74.69</td>
      <td>7.0</td>
      <td>26.1415</td>
      <td>548.9715</td>
      <td>2019-01-05</td>
      <td>13:08</td>
      <td>Ewallet</td>
      <td>522.83</td>
      <td>4.761905</td>
      <td>26.1415</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>226-31-3081</td>
      <td>C</td>
      <td>Naypyitaw</td>
      <td>Normal</td>
      <td>Female</td>
      <td>Electronic accessories</td>
      <td>15.28</td>
      <td>5.0</td>
      <td>3.8200</td>
      <td>80.2200</td>
      <td>2019-03-08</td>
      <td>10:29</td>
      <td>Cash</td>
      <td>76.40</td>
      <td>4.761905</td>
      <td>3.8200</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>631-41-3108</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Home and lifestyle</td>
      <td>46.33</td>
      <td>7.0</td>
      <td>16.2155</td>
      <td>340.5255</td>
      <td>2019-03-03</td>
      <td>13:23</td>
      <td>Credit card</td>
      <td>324.31</td>
      <td>4.761905</td>
      <td>16.2155</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>123-19-1176</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Male</td>
      <td>Health and beauty</td>
      <td>58.22</td>
      <td>8.0</td>
      <td>23.2880</td>
      <td>489.0480</td>
      <td>2019-01-27</td>
      <td>20:33</td>
      <td>Ewallet</td>
      <td>465.76</td>
      <td>4.761905</td>
      <td>23.2880</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>373-73-7910</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Sports and travel</td>
      <td>86.31</td>
      <td>7.0</td>
      <td>30.2085</td>
      <td>634.3785</td>
      <td>2019-02-08</td>
      <td>10:37</td>
      <td>Ewallet</td>
      <td>604.17</td>
      <td>4.761905</td>
      <td>30.2085</td>
      <td>5.3</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Invoice-Id-is-of-no-use-for-us,-so-lets-make-Date-as-our-index-column">Invoice Id is of no use for us, so lets make Date as our index column<a class="anchor-link" href="#Invoice-Id-is-of-no-use-for-us,-so-lets-make-Date-as-our-index-column">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[10]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#make a copy of dataset, in case something goes wrong</span>
<span class="c1">#Copying is not recommended for large datasets</span>

<span class="n">df2</span><span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="n">df</span><span class="o">.</span><span class="n">set_index</span><span class="p">(</span><span class="s2">&quot;Date&quot;</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-see-how-our-dataset-looks-now..">Lets see how our dataset looks now..<a class="anchor-link" href="#Lets-see-how-our-dataset-looks-now..">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[11]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[11]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Invoice ID</th>
      <th>Branch</th>
      <th>City</th>
      <th>Customer type</th>
      <th>Gender</th>
      <th>Product line</th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>Time</th>
      <th>Payment</th>
      <th>cogs</th>
      <th>gross margin percentage</th>
      <th>gross income</th>
      <th>Rating</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-05</th>
      <td>750-67-8428</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Female</td>
      <td>Health and beauty</td>
      <td>74.69</td>
      <td>7.0</td>
      <td>26.1415</td>
      <td>548.9715</td>
      <td>13:08</td>
      <td>Ewallet</td>
      <td>522.83</td>
      <td>4.761905</td>
      <td>26.1415</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>2019-03-08</th>
      <td>226-31-3081</td>
      <td>C</td>
      <td>Naypyitaw</td>
      <td>Normal</td>
      <td>Female</td>
      <td>Electronic accessories</td>
      <td>15.28</td>
      <td>5.0</td>
      <td>3.8200</td>
      <td>80.2200</td>
      <td>10:29</td>
      <td>Cash</td>
      <td>76.40</td>
      <td>4.761905</td>
      <td>3.8200</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>2019-03-03</th>
      <td>631-41-3108</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Home and lifestyle</td>
      <td>46.33</td>
      <td>7.0</td>
      <td>16.2155</td>
      <td>340.5255</td>
      <td>13:23</td>
      <td>Credit card</td>
      <td>324.31</td>
      <td>4.761905</td>
      <td>16.2155</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>2019-01-27</th>
      <td>123-19-1176</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Male</td>
      <td>Health and beauty</td>
      <td>58.22</td>
      <td>8.0</td>
      <td>23.2880</td>
      <td>489.0480</td>
      <td>20:33</td>
      <td>Ewallet</td>
      <td>465.76</td>
      <td>4.761905</td>
      <td>23.2880</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>2019-02-08</th>
      <td>373-73-7910</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Sports and travel</td>
      <td>86.31</td>
      <td>7.0</td>
      <td>30.2085</td>
      <td>634.3785</td>
      <td>10:37</td>
      <td>Ewallet</td>
      <td>604.17</td>
      <td>4.761905</td>
      <td>30.2085</td>
      <td>5.3</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-check-some-Statistics-for-our-data">Lets check some Statistics for our data<a class="anchor-link" href="#Lets-check-some-Statistics-for-our-data">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[12]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[12]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>cogs</th>
      <th>gross margin percentage</th>
      <th>gross income</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>996.000000</td>
      <td>983.000000</td>
      <td>1003.000000</td>
      <td>1003.000000</td>
      <td>1003.000000</td>
      <td>1.003000e+03</td>
      <td>1003.000000</td>
      <td>1003.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>55.764568</td>
      <td>5.501526</td>
      <td>15.400368</td>
      <td>323.407726</td>
      <td>308.007358</td>
      <td>4.761905e+00</td>
      <td>15.400368</td>
      <td>6.972682</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26.510165</td>
      <td>2.924673</td>
      <td>11.715192</td>
      <td>246.019028</td>
      <td>234.303836</td>
      <td>8.886215e-16</td>
      <td>11.715192</td>
      <td>1.717647</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.080000</td>
      <td>1.000000</td>
      <td>0.508500</td>
      <td>10.678500</td>
      <td>10.170000</td>
      <td>4.761905e+00</td>
      <td>0.508500</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>33.125000</td>
      <td>3.000000</td>
      <td>5.894750</td>
      <td>123.789750</td>
      <td>117.895000</td>
      <td>4.761905e+00</td>
      <td>5.894750</td>
      <td>5.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.420000</td>
      <td>5.000000</td>
      <td>12.096000</td>
      <td>254.016000</td>
      <td>241.920000</td>
      <td>4.761905e+00</td>
      <td>12.096000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>78.085000</td>
      <td>8.000000</td>
      <td>22.539500</td>
      <td>473.329500</td>
      <td>450.790000</td>
      <td>4.761905e+00</td>
      <td>22.539500</td>
      <td>8.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>99.960000</td>
      <td>10.000000</td>
      <td>49.650000</td>
      <td>1042.650000</td>
      <td>993.000000</td>
      <td>4.761905e+00</td>
      <td>49.650000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Lets-see-if-data-cleaning-is-required">Lets see if data cleaning is required<a class="anchor-link" href="#Lets-see-if-data-cleaning-is-required">&#182;</a></h3>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="check-for-duplicated-rows-in-the-dataset">check for duplicated rows in the dataset<a class="anchor-link" href="#check-for-duplicated-rows-in-the-dataset">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[13]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">duplicated</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[13]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>3</pre>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[14]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="o">.</span><span class="n">duplicated</span><span class="p">()</span><span class="o">==</span><span class="kc">True</span><span class="p">]</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[14]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Invoice ID</th>
      <th>Branch</th>
      <th>City</th>
      <th>Customer type</th>
      <th>Gender</th>
      <th>Product line</th>
      <th>Unit price</th>
      <th>Quantity</th>
      <th>Tax 5%</th>
      <th>Total</th>
      <th>Time</th>
      <th>Payment</th>
      <th>cogs</th>
      <th>gross margin percentage</th>
      <th>gross income</th>
      <th>Rating</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-18</th>
      <td>849-09-3807</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Member</td>
      <td>Female</td>
      <td>Fashion accessories</td>
      <td>88.34</td>
      <td>7.0</td>
      <td>30.919</td>
      <td>649.299</td>
      <td>13:28</td>
      <td>Cash</td>
      <td>618.38</td>
      <td>4.761905</td>
      <td>30.919</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>2019-03-10</th>
      <td>745-74-0715</td>
      <td>A</td>
      <td>Yangon</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Electronic accessories</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.803</td>
      <td>121.863</td>
      <td>20:46</td>
      <td>Ewallet</td>
      <td>116.06</td>
      <td>4.761905</td>
      <td>5.803</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>2019-01-26</th>
      <td>452-04-8808</td>
      <td>B</td>
      <td>Mandalay</td>
      <td>Normal</td>
      <td>Male</td>
      <td>Electronic accessories</td>
      <td>87.08</td>
      <td>NaN</td>
      <td>30.478</td>
      <td>640.038</td>
      <td>15:17</td>
      <td>Cash</td>
      <td>609.56</td>
      <td>4.761905</td>
      <td>30.478</td>
      <td>5.5</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-remove-the-duplicated-rows">Lets remove the duplicated rows<a class="anchor-link" href="#Lets-remove-the-duplicated-rows">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[15]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">drop_duplicates</span><span class="p">(</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Check-for-duplicated-row-count-again">Check for duplicated row count again<a class="anchor-link" href="#Check-for-duplicated-row-count-again">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[16]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">duplicated</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[16]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>0</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="check-columnwise-Null-values">check columnwise Null values<a class="anchor-link" href="#check-columnwise-Null-values">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[17]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[17]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>Invoice ID                  0
Branch                      0
City                        0
Customer type              79
Gender                      0
Product line               43
Unit price                  6
Quantity                   19
Tax 5%                      0
Total                       0
Time                        0
Payment                     0
cogs                        0
gross margin percentage     0
gross income                0
Rating                      0
dtype: int64</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Graphically,-null-values-can-be-seen-using-heatmap">Graphically, null values can be seen using heatmap<a class="anchor-link" href="#Graphically,-null-values-can-be-seen-using-heatmap">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[18]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">isnull</span><span class="p">())</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[18]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:ylabel=&#39;Date&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Filling-null-values">Filling null values<a class="anchor-link" href="#Filling-null-values">&#182;</a></h5>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="For-numerical-variables,-we-use-the-mean(choice-may-vary-dependent-on-the-spread-of-data)">For numerical variables, we use the mean(choice may vary dependent on the spread of data)<a class="anchor-link" href="#For-numerical-variables,-we-use-the-mean(choice-may-vary-dependent-on-the-spread-of-data)">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[19]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">mean</span><span class="p">(),</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="For-categorical-values,-we-use-mode-to-fill-the-most-frequest-occurance">For categorical values, we use mode to fill the most frequest occurance<a class="anchor-link" href="#For-categorical-values,-we-use-mode-to-fill-the-most-frequest-occurance">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[20]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">mode</span><span class="p">()</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-visualize-using-heatmap-again">Lets visualize using heatmap again<a class="anchor-link" href="#Lets-visualize-using-heatmap-again">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[21]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">isnull</span><span class="p">())</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[21]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:ylabel=&#39;Date&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-start-with-taking-a-look-at-the-branchwise-transactions">Lets start with taking a look at the branchwise transactions<a class="anchor-link" href="#Lets-start-with-taking-a-look-at-the-branchwise-transactions">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[22]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Branch</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[22]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:xlabel=&#39;Branch&#39;, ylabel=&#39;count&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="now,-lets-see-the-transaction-methods-used-for-sales">now, lets see the transaction methods used for sales<a class="anchor-link" href="#now,-lets-see-the-transaction-methods-used-for-sales">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[23]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Payment</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[23]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:xlabel=&#39;Payment&#39;, ylabel=&#39;count&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYUAAAEGCAYAAACKB4k+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAVg0lEQVR4nO3dfbRldX3f8fdHnqtWINyQkaEZqqMGUQe8EpTIQjQG7bID1hhooqCkY1JwxSSmRVcb0UqiNUqrtthRkCGxIgYoo6VVRBvQ8uAdHIYBYpwoFsYRrqL4TGD49o/zu3uOM3dm7sDsc4a579dae529f/vhfO8965zP2U+/k6pCkiSAx427AEnSrsNQkCR1DAVJUsdQkCR1DAVJUmfPcRfwaBx00EG1aNGicZchSY8pq1at+k5VTcw27zEdCosWLWJqamrcZUjSY0qSb25tnoePJEkdQ0GS1DEUJEmd3kIhyb5JbkpyS5Lbkry9tV+U5BtJVrdhSWtPkvcnWZdkTZKj+qpNkjS7Pk80PwCcUFU/SrIX8MUk/6vN+5Oq+uvNln8ZsLgNvwqc3x4lSSPS255CDfyoTe7Vhm31vrcUuLitdwOwf5IFfdUnSdpSr+cUkuyRZDVwL3B1Vd3YZp3bDhGdl2Sf1nYIcNfQ6ne3ts23uSzJVJKp6enpPsuXpHmn11Coqo1VtQRYCByd5AjgLcAzgOcBBwL/dge3ubyqJqtqcmJi1nsvJEmP0EiuPqqq7wNfAE6sqg3tENEDwEeBo9ti64FDh1Zb2NokSSPS24nmJBPAg1X1/ST7Ab8OvDvJgqrakCTAScDatspK4KwklzA4wXx/VW3YWfU8908u3lmb0jases9re9nu/3vHs3rZrjb5J39667hL0C6gz6uPFgArkuzBYI/k0qr6dJLPt8AIsBr4vbb8VcDLgXXAT4DX9VibJGkWvYVCVa0Bjpyl/YStLF/AmX3VI0naPu9oliR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1+vw5TkkC4NgPHDvuEnZ7X3rjl3bKdtxTkCR1DAVJUsdQkCR1eguFJPsmuSnJLUluS/L21n5YkhuTrEvyiSR7t/Z92vS6Nn9RX7VJkmbX557CA8AJVfUcYAlwYpJjgHcD51XVU4HvAWe05c8Avtfaz2vLSZJGqLdQqIEftcm92lDACcBft/YVwEltfGmbps1/cZL0VZ8kaUu9nlNIskeS1cC9wNXA3wPfr6qH2iJ3A4e08UOAuwDa/PuBX5hlm8uSTCWZmp6e7rN8SZp3eg2FqtpYVUuAhcDRwDN2wjaXV9VkVU1OTEw82s1JkoaM5Oqjqvo+8AXg+cD+SWZumlsIrG/j64FDAdr8JwHfHUV9kqSBPq8+mkiyfxvfD/h14A4G4fCqtthpwJVtfGWbps3/fFVVX/VJkrbUZzcXC4AVSfZgED6XVtWnk9wOXJLkncBXgAva8hcAf5lkHXAfcEqPtUmSZtFbKFTVGuDIWdq/zuD8wubtPwN+s696JEnb5x3NkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqROb6GQ5NAkX0hye5LbkvxBaz8nyfokq9vw8qF13pJkXZKvJvmNvmqTJM1uzx63/RDwx1V1c5InAquSXN3mnVdVfzG8cJLDgVOAZwJPBj6X5GlVtbHHGiVJQ3rbU6iqDVV1cxv/IXAHcMg2VlkKXFJVD1TVN4B1wNF91SdJ2tJIzikkWQQcCdzYms5KsibJhUkOaG2HAHcNrXY3s4RIkmVJppJMTU9P91m2JM07vYdCkicAlwFvqqofAOcDTwGWABuA9+7I9qpqeVVNVtXkxMTEzi5Xkua1XkMhyV4MAuFjVXU5QFXdU1Ubq+ph4MNsOkS0Hjh0aPWFrU2SNCJ9Xn0U4ALgjqp631D7gqHFTgbWtvGVwClJ9klyGLAYuKmv+iRJW+rz6qNjgdcAtyZZ3dreCpyaZAlQwJ3AGwCq6rYklwK3M7hy6UyvPJKk0eotFKrqi0BmmXXVNtY5Fzi3r5okSdvmHc2SpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpI6hIEnqGAqSpE5voZDk0CRfSHJ7ktuS/EFrPzDJ1Um+1h4PaO1J8v4k65KsSXJUX7VJkmbX557CQ8AfV9XhwDHAmUkOB84GrqmqxcA1bRrgZcDiNiwDzu+xNknSLHoLharaUFU3t/EfAncAhwBLgRVtsRXASW18KXBxDdwA7J9kQV/1SZK2NKdQSHLNXNq2sf4i4EjgRuDgqtrQZn0bOLiNHwLcNbTa3a1t820tSzKVZGp6enquJUiS5mCboZBk3yQHAgclOaCdDziwfchv8YG9lW08AbgMeFNV/WB4XlUVUDtScFUtr6rJqpqcmJjYkVUlSdux53bmvwF4E/BkYBWQ1v4D4IPb23iSvRgEwseq6vLWfE+SBVW1oR0eure1rwcOHVp9YWuTJI3INvcUquo/V9VhwJur6p9W1WFteE5VbTMUkgS4ALijqt43NGslcFobPw24cqj9te0qpGOA+4cOM0mSRmB7ewoAVNUHkrwAWDS8TlVdvI3VjgVeA9yaZHVreyvwLuDSJGcA3wRe3eZdBbwcWAf8BHjdnP8KSdJOMadQSPKXwFOA1cDG1lzAVkOhqr7IpsNNm3vxLMsXcOZc6pEk9WNOoQBMAoe3D25J0m5qrvcprAV+qc9CJEnjN9c9hYOA25PcBDww01hV/7yXqiRJYzHXUDinzyIkSbuGuV599Dd9FyJJGr+5Xn30Qzbdebw3sBfw46r6x30VJkkavbnuKTxxZrzdlLaUQc+nkqTdyA73ktp6Mf0fwG/s/HIkSeM018NHrxyafByD+xZ+1ktFkqSxmevVR68YGn8IuJPBISRJ0m5krucU7IdIkuaBuf7IzsIkVyS5tw2XJVnYd3GSpNGa64nmjzLo2vrJbfhUa5Mk7UbmGgoTVfXRqnqoDRcB/uyZJO1m5hoK303yO0n2aMPvAN/tszBJ0ujNNRRez+DHcL4NbABeBZzeU02SpDGZ6yWp7wBOq6rvASQ5EPgLBmEhSdpNzHVP4dkzgQBQVfcBR/ZTkiRpXOYaCo9LcsDMRNtTmOtehiTpMWKuH+zvBa5P8sk2/ZvAuf2UJEkal7ne0XxxkinghNb0yqq6vb+yJEnjMOdeUqvq9qr6YBu2GwhJLmx3P68dajsnyfokq9vw8qF5b0myLslXk9gDqySNwQ53nb0DLgJOnKX9vKpa0oarAJIcDpwCPLOt81+T7NFjbZKkWfQWClV1LXDfHBdfClxSVQ9U1TeAdcDRfdUmSZpdn3sKW3NWkjXt8NLMFU2HAHcNLXN3a9tCkmVJppJMTU9P912rJM0row6F84GnAEsY3Bn93h3dQFUtr6rJqpqcmLD7JUnamUYaClV1T1VtrKqHgQ+z6RDReuDQoUUXtjZJ0giNNBSSLBiaPBmYuTJpJXBKkn2SHAYsBm4aZW2SpB7vSk7yceB44KAkdwNvA45PsgQoBj/p+QaAqrotyaXA7Qx+7vPMqtrYV22SpNn1FgpVdeoszRdsY/lz8S5pSRqrcVx9JEnaRRkKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6hgKkqSOoSBJ6vQWCkkuTHJvkrVDbQcmuTrJ19rjAa09Sd6fZF2SNUmO6qsuSdLW9bmncBFw4mZtZwPXVNVi4Jo2DfAyYHEblgHn91iXJGkreguFqroWuG+z5qXAija+AjhpqP3iGrgB2D/Jgr5qkyTNbtTnFA6uqg1t/NvAwW38EOCuoeXubm1bSLIsyVSSqenp6f4qlaR5aGwnmquqgHoE6y2vqsmqmpyYmOihMkmav0YdCvfMHBZqj/e29vXAoUPLLWxtkqQRGnUorAROa+OnAVcOtb+2XYV0DHD/0GEmSdKI7NnXhpN8HDgeOCjJ3cDbgHcBlyY5A/gm8Oq2+FXAy4F1wE+A1/VVlyRp63oLhao6dSuzXjzLsgWc2VctkqS58Y5mSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVLHUJAkdQwFSVJnz3E8aZI7gR8CG4GHqmoyyYHAJ4BFwJ3Aq6vqe+OoT5Lmq3HuKbyoqpZU1WSbPhu4pqoWA9e0aUnSCO1Kh4+WAiva+ArgpPGVIknz07hCoYDPJlmVZFlrO7iqNrTxbwMHz7ZikmVJppJMTU9Pj6JWSZo3xnJOAfi1qlqf5BeBq5P87fDMqqokNduKVbUcWA4wOTk56zKSpEdmLHsKVbW+Pd4LXAEcDdyTZAFAe7x3HLVJ0nw28lBI8vgkT5wZB14KrAVWAqe1xU4Drhx1bZI0343j8NHBwBVJZp7/v1fV/07yZeDSJGcA3wRePYbaJGleG3koVNXXgefM0v5d4MWjrkeStMmudEmqJGnMDAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1DAVJUsdQkCR1drlQSHJikq8mWZfk7HHXI0nzyS4VCkn2AP4L8DLgcODUJIePtypJmj92qVAAjgbWVdXXq+ofgEuApWOuSZLmjVTVuGvoJHkVcGJV/W6bfg3wq1V11tAyy4BlbfLpwFdHXujoHAR8Z9xF6BHz9Xvs2t1fu1+uqonZZuw56koerapaDiwfdx2jkGSqqibHXYceGV+/x675/NrtaoeP1gOHDk0vbG2SpBHY1ULhy8DiJIcl2Rs4BVg55pokad7YpQ4fVdVDSc4CPgPsAVxYVbeNuaxxmheHyXZjvn6PXfP2tdulTjRLksZrVzt8JEkaI0NBktQxFHqSZGOS1UPDo+qyI8k5Sd7cxi9q93Rsa/nTkzz50TyntpTkl5JckuTvk6xKclWSp+3gNu5MclBfNe7Odsb/f7Ptde+lJB+Z6UEhyVt3Vs3bef7/k2SXuvR1lzrRvJv5aVUtGePznw6sBb41xhp2K0kCXAGsqKpTWttzgIOBvxtnbfPBXP//Sfasqod2dPszN802bwX+7NFV/PMeaV2j5p7CCCV5XpLL2/jSJD9NsneSfZN8vbX/qyRfTnJLksuS/KPtbPO5Sf6mfWv6TJIF7ZvPJPCxtpeyX/9/3bzwIuDBqvrQTENV3QJ8Jck1SW5OcmuSpQBJHp/kf7bXcm2S3xra1huHln/GiP+Ox6pZ//9VdV2S45Ncl2QlcHuSPZK8p72X1iR5AwyCJckHW6ebnwN+cWZbM9/ak7wL2K+9dz62eRGt086b2+t6TWs7Osn1Sb6S5P8meXprPz3JyiSfB65Jsl/b07kjyRXALvfedE+hP/slWT00/efAZcCSNv1CBt/kn8fgdbixtV9eVR8GSPJO4AzgA7M9QZK92rylVTXdPnTOrarXt0t731xVUzv1r5rfjgBWzdL+M+DkqvpBOyx0Q/twOhH4VlX9M4AkTxpa5ztVdVSSfw28GfjdLbaqzW3t/z/jKOCIqvpGBt3h3F9Vz0uyD/ClJJ8FjmTQPc7hDPYwbgcuHN5IVZ2d5KzZ9vSTTAAfBo5rz3Ngm/W3wAvbZfUvYbCX8S+G6np2Vd2X5I+An1TVryR5NnDzI/lH9MlQ6M+sh4/asdBfYdD53/uA4xjck3FdW+SIFgb7A09gcM/G1jydwRvl6sGeNXsAG3ZS/Zq7AH+W5DjgYeAQBh84twLvTfJu4NNVdd3QOpe3x1XAK0dZ7G7spqr6Rht/KfDsoXNvTwIWM3i/fbyqNgLfat/gd8QxwLUzz1NV9w1tf0WSxUABew2tc/XQcscB72/rrkmyZgefv3cePhq9axl0Df4g8Dng19ow84FxEXBWVT0LeDuw7za2FeC2qlrShmdV1Ut7q1y3Ac+dpf23gQngue2LwD3AvlX1dwy+Jd4KvDPJnw6t80B73IhfzuZqa///GT8eGg/wxqH3xmFV9dkea/sPwBeq6gjgFfz8+/bHs6+yazIURu864E3A9VU1DfwCg2/8a9v8JwIb2qGh397Otr4KTCR5PgwOJyV5Zpv3w7Yt7TyfB/ZphyYAaIcAfhm4t6oeTPKiNk0GV3/9pKr+CngPg4DQIzfr/z/JC2dZ9jPA77f3EUmeluTxDL6U/VY757CAwXmK2Tw4s+5mbgCOS3JY2+7M4aMnsamfttO38TdcC/zLtu4RwLO3sexYGAr9mTlRNTO8q7XfyODQwrVteg1wa226tfzft2W+xOA45Va135x4FfDuJLcAq4EXtNkXAR/yRPPO016jk4GXtMOAtzE4V3QVMJnkVuC1bHrdngXc1M4tvQ145+ir3n1s4///7VkW/wiD8wU3J1kL/DcGe2RXAF9r8y4Grt/K0y0H1mx+orl9kVsGXN7ec59os/4j8OdJvsK29/zOB56Q5A7gHWz7HMlY2M2FJKnjnoIkqWMoSJI6hoIkqWMoSJI6hoIkqWMoSPxcr7Zrk3wy2+lzapRavz4v2P6S0qNnKEgDP213vh4B/APwe+MuaMjxbLr/ROqVoSBt6TrgqUlekeTG1vPl55IcnORxSb7WOkajTa9LMpFB3/znJ7khydfbN/wLW4+YF81sPMlLW4+aN7e9kie09juTvD1DvacmWcQgoP6w7cnMdveutNMYCtKQJHsy6JvqVuCLwDFVdSRwCfBvquph4K/Y1AXJS4Bb2p2uAAcAzwf+EFgJnAc8E3hWkiWtF9V/B7ykqo4CpoA/GirhO639fAa93N4JfAg4r+3JDHeqJ+10dsQlDQx3dX4dcAGDPqk+0frI2RuY6YHzQuBK4D8Brwc+OrSdT1VVtS4v7qmqWwFalwyLgIUMum3+UuvZdm9+vqsFe0/VWBkK0sAWXZ0n+QDwvqpameR44ByAqroryT1JTmDQBfpwx4UzvZ8+PDQ+M70ng15Rr66qU7dSh72naqw8fCRt3XDPl6dtNu8jDA4jfbL1zT9XNwDHJnkqdL/Otr3fGLbHW42MoSBt3TnAJ5OsAr6z2byVDH4E6aObr7Qt7dzD6cDH2w+sXA9s7+c4PwWc7IlmjYK9pEqPQJJJBid//ZDWbsVjltIOSnI28Pts/0eQpMcc9xQkSR3PKUiSOoaCJKljKEiSOoaCJKljKEiSOv8fNTchuclzbGUAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-check-the-distribution-of-customer-ratings...">Lets check the distribution of customer ratings...<a class="anchor-link" href="#Lets-check-the-distribution-of-customer-ratings...">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[24]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#plotting a hostogram</span>
<span class="n">plt</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Rating</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXcAAAD4CAYAAAAXUaZHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAANh0lEQVR4nO3dW4xd5XmH8edfpohA23CaWsSmtSUQKUIikBEhpUUpTitOwlYVIVAPFrLqXtCEJJWC0xtuQYqaULVCsoDEUQkNdYiM0oqCHNKoF7U6BlQODsUlHOzaeKIAacMFuHl7MSvVxB3Xnr32sGZ/en6Stfda+7DezYhnlr/Ze5yqQpLUlp8begBJ0vgZd0lqkHGXpAYZd0lqkHGXpAZNDT0AwNlnn11r164degxJmih79uz5QVVNL3bbioj72rVrmZ2dHXoMSZooSV451m0uy0hSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg1bEJ1Sl41m79e8GOe7Ld143yHGlvjxzl6QGeeauJRnqDFrS0njmLkkNMu6S1CCXZaQVyh8iv3eGXG5crv/enrlLUoOMuyQ1yLhLUoOMuyQ1yLhLUoMm/t0yLf6UW5L6Ou6Ze5L7kxxO8uyCfWcmeTzJi93lGd3+JPmLJPuS/GuSS5dzeEnS4k7kzP0rwF8CX12wbyuwq6ruTLK1274duAY4v/vzEeCe7lKaSP66BU2q4565V9V3gR8etXsDsL27vh3YuGD/V2vePwOnJzlnTLNKkk7QqD9QXVVVB7vrh4BV3fXVwGsL7re/2/d/JNmSZDbJ7Nzc3IhjSJIW0/vdMlVVQI3wuG1VNVNVM9PT033HkCQtMOq7ZV5Pck5VHeyWXQ53+w8A5y6435pun6QJ4TvQ2jDqmfsjwKbu+iZg54L9f9i9a+Zy4K0FyzeSpPfIcc/ckzwIfAw4O8l+4A7gTuChJJuBV4Abu7v/PXAtsA94G7hlGWaWJB3HceNeVTcf46b1i9y3gFv7DiVJ6sdfPyBJDTLuktQg4y5JDTLuktQg4y5JDTLuktQg4y5JDZr4f6xjSEN9TNuPaEs6Hs/cJalBxl2SGmTcJalBrrlLWjH8Zw3HxzN3SWqQcZekBhl3SWqQa+4TyHVJScfjmbskNci4S1KDjLskNci4S1KDjLskNci4S1KDjLskNci4S1KDjLskNci4S1KDjLskNci4S1KDjLskNahX3JN8JslzSZ5N8mCSU5KsS7I7yb4kX09y8riGlSSdmJHjnmQ18ClgpqouAk4CbgLuAr5YVecBbwCbxzGoJOnE9V2WmQLel2QKOBU4CFwF7Ohu3w5s7HkMSdISjRz3qjoAfAF4lfmovwXsAd6sqiPd3fYDqxd7fJItSWaTzM7NzY06hiRpEX2WZc4ANgDrgA8ApwFXn+jjq2pbVc1U1cz09PSoY0iSFtFnWebjwPeraq6q3gUeBq4ATu+WaQDWAAd6zihJWqI+cX8VuDzJqUkCrAeeB54APtHdZxOws9+IkqSl6rPmvpv5H5w+CTzTPdc24Hbgs0n2AWcB941hTknSEkwd/y7HVlV3AHcctfsl4LI+zytJ6sdPqEpSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDWoV9yTnJ5kR5LvJdmb5KNJzkzyeJIXu8szxjWsJOnE9D1zvxt4tKo+CFwM7AW2Aruq6nxgV7ctSXoPjRz3JO8HrgTuA6iqd6rqTWADsL2723ZgY78RJUlL1efMfR0wB3w5yVNJ7k1yGrCqqg529zkErFrswUm2JJlNMjs3N9djDEnS0frEfQq4FLinqi4BfsxRSzBVVUAt9uCq2lZVM1U1Mz093WMMSdLR+sR9P7C/qnZ32zuYj/3rSc4B6C4P9xtRkrRUI8e9qg4BryW5oNu1HngeeATY1O3bBOzsNaEkacmmej7+k8ADSU4GXgJuYf4bxkNJNgOvADf2PIYkaYl6xb2qngZmFrlpfZ/nlST14ydUJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBxl2SGmTcJalBveOe5KQkTyX5Vre9LsnuJPuSfD3Jyf3HlCQtxTjO3G8D9i7Yvgv4YlWdB7wBbB7DMSRJS9Ar7knWANcB93bbAa4CdnR32Q5s7HMMSdLS9T1z/xLwOeAn3fZZwJtVdaTb3g+sXuyBSbYkmU0yOzc313MMSdJCI8c9yfXA4araM8rjq2pbVc1U1cz09PSoY0iSFjHV47FXADckuRY4Bfgl4G7g9CRT3dn7GuBA/zElSUsx8pl7VX2+qtZU1VrgJuDbVfV7wBPAJ7q7bQJ29p5SkrQky/E+99uBzybZx/wa/H3LcAxJ0v+jz7LM/6qq7wDf6a6/BFw2jueVJI3GT6hKUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoOMuyQ1yLhLUoNGjnuSc5M8keT5JM8lua3bf2aSx5O82F2eMb5xJUknos+Z+xHgT6vqQuBy4NYkFwJbgV1VdT6wq9uWJL2HRo57VR2sqie76/8J7AVWAxuA7d3dtgMbe84oSVqisay5J1kLXALsBlZV1cHupkPAqmM8ZkuS2SSzc3Nz4xhDktTpHfckvwB8A/h0Vf1o4W1VVUAt9riq2lZVM1U1Mz093XcMSdICveKe5OeZD/sDVfVwt/v1JOd0t58DHO43oiRpqfq8WybAfcDeqvrzBTc9Amzqrm8Cdo4+niRpFFM9HnsF8AfAM0me7vb9GXAn8FCSzcArwI29JpQkLdnIca+qfwJyjJvXj/q8kqT+/ISqJDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg4y7JDXIuEtSg5Yl7kmuTvJCkn1Jti7HMSRJxzb2uCc5Cfgr4BrgQuDmJBeO+ziSpGNbjjP3y4B9VfVSVb0D/A2wYRmOI0k6hqlleM7VwGsLtvcDHzn6Tkm2AFu6zf9K8sKIxzsb+MGIj11pfC0rTyuvA3wtK1Lu6vVafvVYNyxH3E9IVW0DtvV9niSzVTUzhpEG52tZeVp5HeBrWamW67Usx7LMAeDcBdtrun2SpPfIcsT9X4Dzk6xLcjJwE/DIMhxHknQMY1+WqaojSf4E+AfgJOD+qnpu3MdZoPfSzgria1l5Wnkd4GtZqZbltaSqluN5JUkD8hOqktQg4y5JDZr4uCc5KclTSb419Cx9JHk5yTNJnk4yO/Q8o0pyepIdSb6XZG+Sjw490yiSXNB9LX7650dJPj30XKNK8pkkzyV5NsmDSU4ZeqZRJLmtew3PTdrXI8n9SQ4neXbBvjOTPJ7kxe7yjHEdb+LjDtwG7B16iDH5rar60IS/f/du4NGq+iBwMRP6tamqF7qvxYeADwNvA98cdqrRJFkNfAqYqaqLmH+jw03DTrV0SS4C/oj5T8FfDFyf5Lxhp1qSrwBXH7VvK7Crqs4HdnXbYzHRcU+yBrgOuHfoWQRJ3g9cCdwHUFXvVNWbgw41HuuBf6+qV4YepIcp4H1JpoBTgf8YeJ5R/Bqwu6rerqojwD8CvzvwTCesqr4L/PCo3RuA7d317cDGcR1vouMOfAn4HPCTgecYhwIeS7Kn+9UMk2gdMAd8uVsquzfJaUMPNQY3AQ8OPcSoquoA8AXgVeAg8FZVPTbsVCN5FvjNJGclORW4lp/9wOQkWlVVB7vrh4BV43riiY17kuuBw1W1Z+hZxuQ3qupS5n+b5q1Jrhx6oBFMAZcC91TVJcCPGeNfM4fQfRDvBuBvh55lVN067gbmv/l+ADgtye8PO9XSVdVe4C7gMeBR4Gngv4ecaZxq/n3pY3tv+sTGHbgCuCHJy8z/5smrkvz1sCONrju7oqoOM7+2e9mwE41kP7C/qnZ32zuYj/0kuwZ4sqpeH3qQHj4OfL+q5qrqXeBh4NcHnmkkVXVfVX24qq4E3gD+beiZeno9yTkA3eXhcT3xxMa9qj5fVWuqai3zf23+dlVN3NkIQJLTkvziT68Dv8P8X0EnSlUdAl5LckG3az3w/IAjjcPNTPCSTOdV4PIkpyYJ81+XifxBd5Jf7i5/hfn19q8NO1FvjwCbuuubgJ3jeuLBfiukfsYq4Jvz/98xBXytqh4ddqSRfRJ4oFvOeAm4ZeB5RtZ9o/1t4I+HnqWPqtqdZAfwJHAEeIrJ/fj+N5KcBbwL3DpJP7BP8iDwMeDsJPuBO4A7gYeSbAZeAW4c2/H89QOS1J6JXZaRJB2bcZekBhl3SWqQcZekBhl3SWqQcZekBhl3SWrQ/wAvuhVcA199VAAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Okay,-that-was-a-basic-distribution,-lets-exlpore-a-detailed-one..">Okay, that was a basic distribution, lets exlpore a detailed one..<a class="anchor-link" href="#Okay,-that-was-a-basic-distribution,-lets-exlpore-a-detailed-one..">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[25]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#using distribution plot from seaborn library</span>
<span class="n">sns</span><span class="o">.</span><span class="n">distplot</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Rating</span><span class="p">)</span>

<span class="c1">#lets check some percentile values as well.</span>

<span class="n">plt</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Rating</span><span class="p">),</span> <span class="n">c</span><span class="o">=</span><span class="s2">&quot;r&quot;</span><span class="p">,</span> <span class="n">ls</span><span class="o">=</span><span class="s2">&quot;-.&quot;</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="s2">&quot;Mean&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">percentile</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Rating</span><span class="p">,</span><span class="mi">25</span><span class="p">),</span><span class="n">c</span><span class="o">=</span><span class="s2">&quot;g&quot;</span><span class="p">,</span><span class="n">ls</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="s2">&quot;25th,</span><span class="se">\n</span><span class="s2">75th </span><span class="se">\n</span><span class="s2">percentile&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">percentile</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Rating</span><span class="p">,</span><span class="mi">75</span><span class="p">),</span><span class="n">c</span><span class="o">=</span><span class="s2">&quot;g&quot;</span><span class="p">,</span><span class="n">ls</span><span class="o">=</span><span class="s2">&quot;--&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-check-distributions-of-all-variables-at-once">Lets check distributions of all variables at once<a class="anchor-link" href="#Lets-check-distributions-of-all-variables-at-once">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[26]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">hist</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span><span class="mi">10</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAsYAAAJOCAYAAAC0i8EAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABeq0lEQVR4nO39fbwkdXnn/7/egjc4GAExJwjEwZXoEicqmUWMbnIiRhFvMPs1BJcoGPKbzQaNxkniaLKrMbqLWVExNyYT0cGIKEENREyUEM4ak4gBJQ6CroiDzGRgUAEZNOrg9fuj6mBzOGfOTd/VOef1fDz60d2fqq66qrqv7qurPlWVqkKSJEla7e437gAkSZKkLrAwliRJkrAwliRJkgALY0mSJAmwMJYkSZIAC2NJkiQJsDBeMZL8aZL/MeR57E7yqGHOQ9LwmMOStHcWxh2RpJI8ekbb65K8dyGvr6pfrarfb183mWT7oGOsqv2r6oZBT1dayZKclmRrkm8luTnJnyR56AjmO5XkV3rbenM4yZYkbxh2HNJK0f6xnL59P8m3e56fMqB5bJsx3Y/3DDsuyVfa75GTe9oPSPKZJA8ZRAyrnYWx5pVk33HHIC1HSTYCbwJ+C3gocCywFvh4kvuPMTRJi9T+sdy/qvYHvgo8t6ftvAHOqne6z+hpfxvwXOCZwJ8k2adt/9/AmVV15wBjWLUsjJeJ6a3ASTYm2ZVkZ5KX9AzfkuQNSdYAfwM8oucf5yNmmd6WtvvFpUnuTPJ/kzyyZ3glOSPJl4Av9bQ9un28X5KzktyY5I4kn0yyXzvs2CT/lOT2JP+aZHKoK0fqoCQ/BPwe8LKq+tuq+l5VbQNOAh4F/NeZW21n7u1JsinJl9scvTbJz/cMO63Nuzcnua3dkvSsdtgbgf8M/FH7HfBHbXsleXSSDcApwG+3w/86yW8l+eCMZXh7krOHtY6klSDJMUn+uf3N25nkj5I8oB32U0m+luTw9vnj23x97BJmtaaqrqmqfwW+CzwsyTHAEVV1weCWaHWzMF5efoRmq9OhwOnAHyc5sHeEqroLeBbwbz3/OP9tjumdAvw+cDBwNTDzH+/zgScBR83y2jcDPwn8FHAQ8NvA95McClwCvKFt/03gg0kevqgllZa/nwIeBHyot7GqdgMfBZ4x24tm+DJNgftQmiL7vUkO6Rn+JOCLNDn8B8A5SVJVvwP8A/DS9jvgpTNi2EyT73/QDn8u8F7g+CQHwD17ik4G3rOopZZWn7uB36DJwycDxwG/BlBV/wT8GXBuu/HovcD/qKov7GV65yW5NcnHkzy+p31XW1g/Hvg+cBtwNvDrA1+iVczCeHn5HvD6dsvTR4HdwGP6mN4lVfWJqvoO8DvAk6f/1bb+d1V9o6q+3fuiJPcDfhl4eVXtqKq7q+qf2un8EvDRqvpoVX2/qi4FrgRO6CNOaTk6GPhaVe2ZZdhOYN4/i1X1l1X1b20ufYBm780xPaPcWFV/XlV3A+cChwATSwm2qnYCnwB+oW06vo3/qqVMT1otquqqqvpUVe1p9wr9GfAzPaO8jubP7aeBHcAf72Vyp9B0t3okcDnwsek/q8Cv0hTCm4EXAf8d+DvgQUk+luTyJD9znylqUSyMu+NuYGafw/vTFMPTvj7jR/ZbwP59zPOm6QftVqxvAI+YbfgMB9NsCfvyLMMeCfxCu0vp9iS3A0+l+cGWVpOvAQfP0Uf/kHb4XiV5cZKre3LpcTT5N+3m6QdV9a32YT/fCefS/Lmlvf+LPqYlrQpJfizJR9qD4r4J/C968rSqvgdsocnfs6qq5ppWVf1jVX27qr5VVf8buJ1mrxFVdXVVTVbVk4BraTZQ/S/gnTR7lF4C/EWSDGM5VwsL4+74Ks2/xF5HADcuYVpzJt0M92wdTrI/TdeH3m4Xc03na8C/A/9hlmE3AX9RVQf03NZU1ZkLjElaKf4Z+A7wX3ob21x7FjAF3AU8uGfwj/SM90jgz4GXAg+rqgOAa4CF/ujN9z0w2/C/An4iyeOA53Df7lWS7usdwBeAI6vqh4DX0JOnbRfD1wLvBs5K8sBFTLuYPeffCvxuu0d3HXBlu7X6/ixgb5TmZmHcHR8AfjfJYUnul+TpNEefXriEad1C0yl/vlNCnZDkqe1BAr8PfKqq5tpKfI+q+j7wLuAtSR6RZJ8kT26T/b3Ac5M8s21/UHtA0WFLWA5p2aqqO2i24vxhkuOT3D/JWuACmj+X59H07T8hyUFJfgR4Rc8k1tD8KN4KkOZg28ctIoRbaA7yW/Dwqvp3mu+c9wGfrqqvLmJ+0mr1EOCbwO72oLr/Pj2g3Xq7BTiH5tignTS/t/eR5EeTPCXJA9rfzt+i2fL8jzPG+zngQVX1kbbpK8DTkvw48EDg64NcuNXGwrg7Xg/8E/BJmg71fwCcUlXXLHZCbaf+84Eb2l2w9zkrRet9NP9iv0FzIN0vzTHebH4T2Ar8S/v6NwH3awvrE2n+Md9KswX5t/CzplWoqv6AJhfeDNxJ8wP2YODp7YGyfwH8K7AN+DjNH+Tp114LnEWz5fkWmq1C9/qBnMfZwAvaI+DfPsvwc4Cj2u+Iv+ppP7edl90opIX5TeC/0uT4n9OTxzQHxv0wzQF3RdPd4SVJ/vMs03kIzdbn22j6Ih8PPKuq7il02w1Q/wd4ec/rXgb8KU1/419rjznQEmUvXV20giXZAmyvqt8ddyzSatFu9X098JSubo1N8qM0u4V/pKq+Oe54JGmUvHCDJI1IVb07yR6aU7l1rjBuzzjzSuD9FsWSViMLY0kaoarqZBeFNBcHuoXmgN/jxxyOJI2FXSkkSZIkFnBAVJJ3pbkE8TU9bf8nyReSfC7Jh3tOPk2SVye5PskXkzxzSHFLWgTzWJKk+c27xTjJT9NcYe09VfW4tu0ZwN9X1Z4kbwKoqlclOYrmbAjH0Fwo4u+AH5vvCMmDDz641q5d2++yLMhdd93FmjVrRjKvxTCuhetiTDB/XFddddXXqmos55c0j0eji3F1MSZYvnGNM49HYW953JX3zDjuqyuxdCUO2Hsse83jqpr3RnPhiWvmGPbzwHnt41cDr+4Z9jHgyfNN/yd/8idrVC6//PKRzWsxjGvhuhhT1fxx0ZyAfUE5N4ybeTx8XYyrizFVLd+4xp3Hw77tLY+78p4Zx311JZauxFG191j2lseDOPjul/nBOfsOBT7VM2x723YfSTYAGwAmJiaYmpoaQCjz271798jmtRjGtXBdjAm6G9cCmccD0MW4uhgTGJekbuqrME7yO8AelnDZ0KraDGwGWL9+fU1OTvYTyoJNTU0xqnkthnEtXBdjgu7GNR/zeHC6GFcXYwLjktRNSy6Mk5wGPAc4rt0sDc2VWg7vGe2wtk1SB5nHkiT9wJIu05vkeOC3gedV1bd6Bl0MnJzkgUmOAI4EPt1/mJIGzTyWJOne5t1inOR8YBI4OMl24LU0B+c8ELg0CcCnqupXq+rzSS4ArqXZNXtGec1uaezMY0mS5jdvYVxVL5yl+Zy9jP9G4I39BCWN2tpNl/Q9jS3Hd+MUNbMxj7UarPQ87oJBrONtZz57AJFIw7GkrhSSJEnSSjOI07VJksZgIVvvNq7bw2l7Gc+td5L0A24xliRJklhlW4zXbrpk3q0ne+OWFUmSpJXLLcaSJEkSFsaSJEkSsMq6UkgaH0/zJEnqOrcYS5IkSbjFWGO2lK2IMw+gdCuiJEkaBAtjSVqkuf7QLeasN/6hk6TusSuFJEmShFuMJUnSCPV7IO7GdXuYHEwo0n24xViSJEnCLcbLkqe9kiRJGjy3GEuSJElYGEuSJEnAAgrjJO9KsivJNT1tByW5NMmX2vsD2/YkeXuS65N8LsnRwwxe0sKYx5IkzW8hW4y3AMfPaNsEXFZVRwKXtc8BngUc2d42AO8YTJiS+rQF81ha1pIcnuTyJNcm+XySl7ft/smVBmTewriqPgF8Y0bzicC57eNzgef3tL+nGp8CDkhyyIBilbRE5rG0IuwBNlbVUcCxwBlJjsI/udLALPWsFBNVtbN9fDMw0T4+FLipZ7ztbdtOZkiygSZRmZiYYGpqaomhLNzGdXuY2K+5X4phxrh79+4FT3+p8fda6LwWE9dSLGVZZr6Hg4hvEOt02OtqCEaax4P83I57Xc+1LIv5fvnD8y4aQBzzjzNfTONaj8N4D1d6Hrf5urN9fGeS62hy80S459S+5wJTwKvo+ZMLfCrJAUkO6cl7STP0fbq2qqoktYTXbQY2A6xfv74mJyf7DWVep226hI3r9nDW1qUt9rZTJgcbUI+pqSkWug4WesnZvVnosiwmrqVYyrLMfA8H8b4MYp1uOX7NUNfVMI0ijwf5uR3253I+cy1LP98vwzJvTFvv6nseSzn94zDew9WUx0nWAk8ErqDPP7kL/YO7e/duNq67ezAL0IeJ/cb3h65Xl/5EdSWWrsQBS49lqd/gt0z/62x3se5q23cAh/eMd1jbJql7zGNpGUqyP/BB4BVV9c0k9wxbyp/chf7BnZqa4qxP9v9Hql8b1+3hpA78eRn3H/ReXYmlK3HA0mNZ6unaLgZObR+fClzU0/7itsP/scAd7rKROss8lpaZJPenKYrPq6oPtc23TB8H4J9cqT8LOV3b+cA/A49Jsj3J6cCZwM8l+RLw9PY5wEeBG4DrgT8Hfm0oUUtaFPNYWv7SbBo+B7iuqt7SM8g/udKAzNuVoqpeOMeg42YZt4Az+g1K0mCZx9KK8BTgRcDWJFe3ba+h+VN7QfuH90bgpHbYR4ETaP7kfgt4yUijlZahbh0lomVl7QAOdJEkLUxVfRLIHIP9kysNgJeEliRJklhGW4zdOjlYC12fG9ftGcgpkCRJkrpu2RTGK8VcBakFqCRJ0nhZGEuSxs69gpK6wMJYy54/qJIkaRA8+E6SJEnCLcaL4pZJSZKklcvCWNKq4h9cSdJcLIwlLRvTRa1ncZEkDYN9jCVJkiQsjCVJkiTAwliSJEkCLIwlSZIkwMJYkiRJAiyMJUmSJKDPwjjJbyT5fJJrkpyf5EFJjkhyRZLrk3wgyQMGFaykwTOPJUlqLPk8xkkOBX4dOKqqvp3kAuBk4ATgrVX1/iR/CpwOvGMg0UoaKPNYg7CUi6Z4LmpJXdRvV4p9gf2S7As8GNgJPA24sB1+LvD8PuchabjMY0mS6GOLcVXtSPJm4KvAt4GPA1cBt1fVnna07cChs70+yQZgA8DExARTU1N7nd/GdXv2OnyhJvYb3LQGybgWrosxAezevXvez3HXjDKPB/medfUz0MW4uhgTdDeu5ZjHkgann64UBwInAkcAtwN/CRy/0NdX1WZgM8D69etrcnJyr+MPapfbxnV7OGtr966EbVwL18WYALYcv4b5PsddM8o8HuRu865+BroYVxdjgu7GtRzzWNLg9NOV4unAV6rq1qr6HvAh4CnAAe0uWYDDgB19xihpeMxjSZJa/RTGXwWOTfLgJAGOA64FLgde0I5zKnBRfyFKGiLzWJKk1pIL46q6gubgnM8AW9tpbQZeBbwyyfXAw4BzBhCnpCEwj6XlI8m7kuxKck1P2+uS7EhydXs7oWfYq9tTLn4xyTPHE7W0vPTVwauqXgu8dkbzDcAx/UxX0uiYx9KysQX4I+A9M9rfWlVv7m1IchTNqRd/HHgE8HdJfqyq7h5FoNJy5ZXvJElaBqrqE8A3Fjj6icD7q+o7VfUV4Hr8syvNq3uHBEuSpMV4aZIXA1cCG6vqNppTLH6qZ5y+T7u4e/duNq4b/wbnif3oxCn1unRqv67E0pU4YOmxWBhLkrR8vQP4faDa+7OAX17MBBZ62sWpqSnO+uRd/cQ6EBvX7eGkDpxSb2pqqjOn9utKLF2JA5Yei4WxJEnLVFXdMv04yZ8DH2mf7gAO7xl1RZ12cSmXIZ9p25nPHkAkWmnsYyxJ0jKV5JCepz8PTJ+x4mLg5CQPTHIEcCTw6VHHJy03bjGWJGkZSHI+MAkcnGQ7zdlkJpM8gaYrxTbgvwFU1eeTXEBzXvI9wBmekUKan4WxJEnLQFW9cJbmOc8xXlVvBN44vIiklceuFJIkSRIWxpIkSRJgYSxJkiQBFsaSJEkSYGEsSZIkARbGkiRJEmBhLEmSJAEWxpIkSRJgYSxJkiQBfRbGSQ5IcmGSLyS5LsmTkxyU5NIkX2rvDxxUsJIGzzyWJKnR7xbjs4G/rarHAo8HrgM2AZdV1ZHAZe1zSd1lHkuSRB+FcZKHAj9Ne532qvpuVd0OnAic2452LvD8/kKUNCzmsSRJP7BvH689ArgVeHeSxwNXAS8HJqpqZzvOzcDEbC9OsgHYADAxMcHU1NReZ7Zx3Z4+Qv2Bif0GN61BMq6F62JMALt37573c9xBI8vjQb5nXf0MdDGuLsYE3Y1rmeaxpAHppzDeFzgaeFlVXZHkbGbsbq2qSlKzvbiqNgObAdavX1+Tk5N7ndlpmy7pI9Qf2LhuD2dt7Wexh8O4Fq6LMQFsOX4N832OO2hkeTyoHIbufga6GFcXY4LuxrVM81jSgPTTx3g7sL2qrmifX0jzA3tLkkMA2vtd/YUoaYjMY0mSWksujKvqZuCmJI9pm44DrgUuBk5t204FLuorQklDYx5LkvQD/e7HehlwXpIHADcAL6Epti9IcjpwI3BSn/OQNFzmsSRJ9FkYV9XVwPpZBh3Xz3QljY55LElSo3tHPkiSJA3Z2j4PCN64bg+TgwlFHeIloSVJkiQsjCVJkiTAwliSpGUhybuS7EpyTU/bQUkuTfKl9v7Atj1J3p7k+iSfS3L0+CKXlg8LY0mSloctwPEz2jYBl1XVkcBl/OACPc8CjmxvG4B3jChGaVmzMJYkaRmoqk8A35jRfCJwbvv4XOD5Pe3vqcangAOmL9ojaW6elUKSpOVroqp2to9vBibax4cCN/WMt71t28kMSTbQbFVmYmKCqampWWe0e/duNq67ezBR92Fiv+aMEOM2sR9zrqtR2717dydi6UocsPRYLIwlSVoBqqqS1BJetxnYDLB+/fqanJycdbypqSnO+uRdfcU4CBvX7eGsreMvXzau28NJc6yrUZuammKu9201xgFLj8WuFJIkLV+3THeRaO93te07gMN7xjusbZO0FxbGkiQtXxcDp7aPTwUu6ml/cXt2imOBO3q6XEiaw/j3RUiSpHklOR+YBA5Osh14LXAmcEGS04EbgZPa0T8KnABcD3wLeMnIA5aWIQtjSZKWgap64RyDjptl3ALOGG5E0spjYSxJkrQEazdd0vc0tp357AFEokGxj7EkSZKEhbEkSZIEWBhLkiRJwAAK4yT7JPlsko+0z49IckWS65N8IMkD+g9T0jCZx5IkDWaL8cuB63qevwl4a1U9GrgNOH0A85A0XOaxJGnV66swTnIY8Gzgne3zAE8DLmxHORd4fj/zkDRc5rEkSY1+T9f2NuC3gYe0zx8G3F5Ve9rn24FDZ3thkg3ABoCJiQmmpqb2OqON6/bsdfhCTew3uGkNknEtXBdjAti9e/e8n+OOehsjyONBvmdd/Qx0Ma4uxgTdjWsZ57GkAVhyYZzkOcCuqroqyeRiX19Vm4HNAOvXr6/Jyb1P4rQBnCsQmi/is7Z27/TNxrVwXYwJYMvxa5jvc9w1o8zjQeUwdPcz0MW4uhgTdDeu5ZjHkgann2+lpwDPS3IC8CDgh4CzgQOS7NtubToM2NF/mJKGxDyWJKm15D7GVfXqqjqsqtYCJwN/X1WnAJcDL2hHOxW4qO8oJQ2FeSxJ0g8M4zzGrwJemeR6mr6K5wxhHpKGyzyWJK06A+ngVVVTwFT7+AbgmEFMV9LomMeSpNXOK99JkiRJDGiLsSRJkhZv7QDO2LPl+DUDiETgFmNJkiQJsDCWJEmSAAtjSZIkCbAwliRJkgALY0mSJAmwMJYkSZIAC2NJkiQJ8DzGkiQte0m2AXcCdwN7qmp9koOADwBrgW3ASVV127hilJYDtxhLkrQy/GxVPaGq1rfPNwGXVdWRwGXtc0l7YWEsSdLKdCJwbvv4XOD54wtFWh7sSiFJ0vJXwMeTFPBnVbUZmKiqne3wm4GJ2V6YZAOwAWBiYoKpqalZZ7B79242rrt70HEv2sR+sHHdnnGH0Zk4oHlv5nrfFmrrjjv6juOIh+7TdxyDstR1YmEsSdLy99Sq2pHkh4FLk3yhd2BVVVs030dbRG8GWL9+fU1OTs46g6mpKc765F2DjXoJNq7bw1lbx1++dCUOgC3Hr2Gu922hTtt0SSfiGJSpqaklxWJXCkmSlrmq2tHe7wI+DBwD3JLkEID2ftf4IpSWhyX/1UlyOPAeml0zBWyuqrM9ClZaPsxjaflLsga4X1Xd2T5+BvB64GLgVODM9v6i8UWpYdq6446BbPFVf1uM9wAbq+oo4FjgjCRH4VGw0nJiHkvL3wTwyST/CnwauKSq/pamIP65JF8Cnt4+l7QXS95i3Hbo39k+vjPJdcChNEfBTrajnQtMAa/qK0pJQ2EeS8tfVd0APH6W9q8Dx40+Imn5Gkiv8SRrgScCVzDgo2CnDerIzy4dRdrLuBauizHBYI4KHqdh5/Eg37Oufga6GFcXY4LuxrXc81hSf/oujJPsD3wQeEVVfTPJPcMGcRTstEH1nenSUaS9jGvhuhgTdOto3MUaRR4Psv9bVz8DXYyrizFBd+NaznksqX99nZUiyf1pfkzPq6oPtc0eBSstI+axJEmNJRfGaTYpnQNcV1Vv6Rk0fRQseBSs1GnmsSRJP9DPfqynAC8Ctia5um17Dc1RrxckOR24ETiprwglDZN5LElSq5+zUnwSyByDPQpWWgbMY0mSfsAr30mSJElYGEuSJEmAhbEkSZIEWBhLkiRJwICufCdJkqTVbeuOO/q6mNO2M589wGiWxi3GkiRJEhbGkiRJEmBhLEmSJAEWxpIkSRLgwXeSJEnqgLV9HLg3rd8D+NxiLEmSJGFhLEmSJAEWxpIkSRJgYSxJkiQBFsaSJEkSYGEsSZIkAUMsjJMcn+SLSa5PsmlY85E0HOawtPyZx9LiDKUwTrIP8MfAs4CjgBcmOWoY85I0eOawtPyZx9LiDWuL8THA9VV1Q1V9F3g/cOKQ5iVp8Mxhafkzj6VFSlUNfqLJC4Djq+pX2ucvAp5UVS/tGWcDsKF9+hjgiwMPZHYHA18b0bwWw7gWrosxwfxxPbKqHj6qYPqxkBxu283je+tiXF2MCZZvXKs5j7vynhnHfXUllq7EAXuPZc48HtsloatqM7B51PNNcmVVrR/1fOdjXAvXxZigu3ENk3l8b12Mq4sxgXF1yULzuCvrxjjuqyuxdCUOWHosw+pKsQM4vOf5YW2bpOXBHJaWP/NYWqRhFcb/AhyZ5IgkDwBOBi4e0rwkDZ45LC1/5rG0SEPpSlFVe5K8FPgYsA/wrqr6/DDmtQQj3+27QMa1cF2MCbob16J1PIehu+u6i3F1MSYwrqEbQh53Zd0Yx311JZauxAFLjGUoB99JkiRJy41XvpMkSZKwMJYkSZKAFV4YJzk8yeVJrk3y+SQvb9sPSnJpki+19weOIbZ9knw2yUfa50ckuaK9bOcH2gMlRh3TAUkuTPKFJNcleXJH1tVvtO/fNUnOT/KgcayvJO9KsivJNT1ts66fNN7exve5JEcPO77VYK6c7oKZOd0Fs+V0B2K6Tz6PKY4F5/NqlzFeVror71NX6on29+/TSf61jeP32vax1BBdqWWSbEuyNcnVSa5s25b03qzowhjYA2ysqqOAY4Ez0lwOcxNwWVUdCVzWPh+1lwPX9Tx/E/DWqno0cBtw+hhiOhv426p6LPD4Nr6xrqskhwK/DqyvqsfRHEByMuNZX1uA42e0zbV+ngUc2d42AO8YQXyrwVw53QUzc7oLZsvpsdlLPo/DFhaez6tWxn9Z6S10433qSj3xHeBpVfV44AnA8UmOZXw1RJdqmZ+tqif0nLt4ae9NVa2aG3AR8HM0V/U5pG07BPjiiOM4rH2TngZ8BAjN1Vn2bYc/GfjYiGN6KPAV2gMye9rHva4OBW4CDqI5i8pHgGeOa30Ba4Fr5ls/wJ8BL5xtPG8DfT8uAn6uA3HcK6fHHU8b06w5PeaYZsvnZ4wxngXl82q+zfx+BV4NvHq1v09dqCeABwOfAZ40jt/ELtUywDbg4BltS3pvVvoW43skWQs8EbgCmKiqne2gm4GJEYfzNuC3ge+3zx8G3F5Ve9rn22l+QEbpCOBW4N3tbpF3JlnDmNdVVe0A3gx8FdgJ3AFcxfjX17S51s90ATBtnDGuSDNyetzexr1zugvmyumxmS2fq+rj44xphnH/NnRRF7/Lxvo+jbueaLsvXA3sAi4Fvsx4fhPfRndqmQI+nuSqNJc4hyW+N6uiME6yP/BB4BVV9c3eYdX8lRjZOeuSPAfYVVVXjWqeC7QvcDTwjqp6InAXM3Y7jHpdAbR9gk6k+ZF/BLCG++5W64RxrJ/Vam85PYZYlm1Oj9ps+Zzkl8YZ01zM5+VhDL/hY68nquruqnoCzRbbY4DHDnueM3Xwe++pVXU0TZefM5L8dO/Axbw3K74wTnJ/mg/xeVX1obb5liSHtMMPofnXNSpPAZ6XZBvwfppdEGcDBySZvuDKOC7buR3YXlXTW98upPlRHee6Ang68JWqurWqvgd8iGYdjnt9TZtr/Xgp1iGZI6fH6T45neS94w0JmDunx2m2fP6pMcfUa9zfd13Uxe+ysbxPXasnqup24HKaLguj/k3sVC3T7o2iqnYBH6b5w7Ck92ZFF8ZJApwDXFdVb+kZdDFwavv4VJq+QiNRVa+uqsOqai3NQSd/X1Wn0Hy4XzCOmNq4bgZuSvKYtuk44FrGuK5aXwWOTfLg9v2cjmus66vHXOvnYuDFaRxLs8t452wT0MLtJafHZo6cHvtW0L3k9DjNls9dOmBx3N93XdTFy0qP/H3qSj2R5OFJDmgf70fTz/k6Rvyb2KVaJsmaJA+Zfgw8A7iGpb43o+gUPa4b8FSaTeefA65ubyfQ9IO5DPgS8HfAQWOKb5L2QB3gUcCngeuBvwQeOIZ4ngBc2a6vvwIO7MK6An4P+EL7Qf8L4IHjWF/A+TT9Ir9HszXu9LnWD81BCH9M0/drK81R+CP/jK2021w5Pe64euK7J6e7cJstpzsQ033yeUxxLDifV/ut/d38f+332e+sxvepK/UE8BPAZ9s4rgH+Z9s+thqCMdcy7Tz/tb19fvozutT3xktCS5IkSazwrhSSJEnSQlkYS5IkSVgYS5IkSYCFsSRJkgRYGEuSJEmAhbEkSZIEWBhLkiRJgIWxJEmSBFgYS5IkSYCFsSRJkgRYGEuSJEmAhbEkSZIEWBhLkiRJgIWxFihJJXn0uOOQJGm5SbI7yaPGHYfmZ2G8zLXJNn37fpJv9zw/ZY7XTCbZPupYJUlajapq/6q6YdxxDEuSbUmePu44BmHfcQeg/lTV/tOPk2wDfqWq/m58EUmStHBJ9q2qPeOOY6mWS/zLJc5xc4vxCpXkgUneluTf2tvb2rY1wN8Aj+jZsvyIJMck+ecktyfZmeSPkjxg3MshrSZJDk/yoSS3Jvl6m4f3S/K7SW5MsivJe5I8tOc1L26HfT3J/+jdctPm9ZVJvpnkliRvGd/SaTVJcnSSzya5M8lfJvlAkje0wyaTbE/yqiQ3A++e6zerHf/gJB9pf5++keQfktyvHfaqJDva+XwxyXFzxLMlyZ8k+Zv2d+8fk/xIO5/bknwhyRN7xt+U5MvtdK9N8vM9w05rX//WJF8HXpfkYUn+us21f0nyhiSf7HnNPd0R21j+OMkl7fSvSPIf5oh7bfvaDe162ZnkN3uG368n1q8nuSDJQTNee3qSrwJ/37b//5Jc17NsR7ftj0jywfb75ytJfr1nPq9rp/2e9nWfT7K+HfYXwI8Cf92u299u2/8yyc1J7kjyiSQ/3jO9+dbXY5Nc2r7fX0xy0l4/cANkYbxy/Q5wLPAE4PHAMcDvVtVdwLOAf2t37exfVf8G3A38BnAw8GTgOODXxhG4tBol2Qf4CHAjsBY4FHg/cFp7+1ngUcD+wB+1rzkK+BPgFOAQ4KHt66adDZxdVT8E/AfggqEviFa9NBtVPgxsAQ4Czgd+fsZoP9IOeySwgTl+s9pxNwLbgYcDE8BrgEryGOClwH+qqocAzwS27SW0k9ppHgx8B/hn4DPt8wuB3j+OXwb+M01O/R7w3iSH9Ax/EnBDG88bgT8G7mqX69T2tjcnt9M9ELi+ncbe/CxwJPAM4FX5QbeFlwHPB34GeARwWxtLr58B/iPwzCS/ALwOeDHwQ8DzgK+3fzT+GvhXmu+Q44BXJHlmz3SeR/OddABwMe33UFW9CPgq8Ny2pviDdvy/aWP+YZr1fF7PtOZcX2k24F0KvK997cnAn7Tfd8NXVd5WyI3mC+Hp7eMvAyf0DHsmsK19PAlsn2darwA+3PO8gEePexm9eVupN5o/pLcC+85ovwz4tZ7njwG+R9MV7n8C5/cMezDw3Z7vgU/Q/PgePO7l87Z6bsBPAzuA9LR9EnhD+3iy/Zw+qGf43n6zXg9cNPM3CHg0sAt4OnD/eWLaAvx5z/OXAdf1PF8H3L6X118NnNg+Pg34as+wfdqcfExP2xuAT/Y8v+c3tI3lnT3DTgC+MMd817avfWxP2x8A57SPrwOO6xl2SM/3w/RrH9Uz/GPAy2eZz5N6l6ltezXw7vbx64C/6xl2FPDtnufbpr935liOA9pYHjrf+gJ+EfiHGa//M+C1o/j8usV45XoEzZanaTe2bbNK8mPtrqqbk3wT+F80/6IljcbhwI113z6As+XyvjRbqh4B3DQ9oKq+BXy9Z9zTgR8DvtDurnzOMAKXZngEsKPaiqZ104xxbq2qf5/xmrl+s/4PzVbVjye5IckmgKq6nmYjzuuAXUnen2TO3znglp7H357lee8xOy9OcnXbfeN24HHc+zexd3keTpOTN80xfDY39zz+Vu+859A7vd5180jgwz1xXkezB3hijtceTvMnZKZH0nSxvL1nWq+ZMZ2ZMT8oyazHqiXZJ8mZbRePb/KDLfkHM//6eiTwpBmxnEKzdXnoLIxXrn+j+XBN+9G2DZp/bTO9A/gCcGQ1u11fA2SoEUrqdRPwo7P80MyWy3toftR3AodND0iyH/Cw6edV9aWqeiHN7sg3ARe2uymlYdoJHJqk9zfk8BnjzPwdmvM3q6rurKqNVfUomt35r0zbl7iq3ldVT21fWzSf874keSTw5zTdNB5WVQcA13Dv38Te+G+lycnDetpmLm+/eqfX+3t+E/Csqjqg5/agqtoxR6w30XSrmukm4CszpvOQqjphgfHNfD//K3Aizdb8h9JsvYZmHc63vm4C/u+MWPavqv++wFj6YmG8cp0P/G6Shyc5mGaX63vbYbcAD0vPATzAQ4BvAruTPBYYyQdQ0j0+TVNQnJlkTZIHJXkKTS7/RpIjkuxPszfnA+2W5QuB5yb5qbZf5+vo+fFO8ktJHl5V3wdub5u/P7pF0ir1zzRbLV+aZN8kJ9L0Gd6bOX+zkjwnyaPbQvuOdtrfT/KYJE9Lc5Dev9Ns9R3E53sNTaF3azv/l9BsMZ5VVd0NfIjmILwHt7+hLx5AHL3+RzvtHwdeAnygbf9T4I1tMU+7/k7cy3TeCfxmkp9M49Htaz8N3JnmYMb92i2+j0vynxYY3y00x0BMewhNP+6v03Tx+l/TAxawvj4C/FiSFyW5f3v7T0n+4wJj6YuF8cr1BuBK4HPAVpqO728AqKov0HwJ3dDupngE8Js0//DupPmn/IHZJippONofi+fS9Jv8Ks3BRr8IvAv4C5r+wl+hKQBe1r7m8+3j99MU1btp+lx+p53s8cDnk+ymORDv5Kr69ogWSatUVX0X+C80XXluB36Jptj5zl5eNudvFs0BXH9H8/n+Z+BPqupy4IHAmcDXaHbz/zBNv9h+478WOKud1y00/Y//cZ6XvZRmy+jNNPl6Pntf3sX6vzTdSS4D3lxVH2/bz6Y5EO7jSe4EPkXTX3hWVfWXNAf6vY/m9/6vgIPa75/n0Bz8+BWadfrOdpkW4n/T/LG5Pc1ZM95D0+VjB3BtG1evOddXVd1Jc5DhyTRbxm+m2RPwwAXG0pfcuwuQJGm5arco307TJeorYw5HukeSK4A/rap3jzuWUUjyJuBHqmq+s1PMN521NIXq/Wc5/mDFGNT6GgS3GEvSMpbkue3uyDXAm2m2tm0bb1Ra7ZL8TJrzBO+b5FTgJ4C/HXdcw9Ked/cn2u4Jx9BsLf/wuOPqqi6vL698J0nL24k0uyJDsyv65HJXoMbvMTTnzV5Dc77fF1TVzvGGNFQPoekO8Aia7hdn0ZxiTrPr7PqyK4UkSZKEXSkkSZIkwMJYkiRJAjrSx/jggw+utWvXzjrsrrvuYs2a8Z6PftwxjHv+xjCYGK666qqvVdXDBxxSZ3Q9j7sUB3QnFuO4r73FsprzeBS68DnoQgzGMdw49prHo7ju9Hy3n/zJn6y5XH755XMOG5VxxzDu+RvDYGIArqwO5Nuwbl3P46ruxFHVnViM4772FstqzuNR6MLnoAsxVBnHTIOMY295bFcKSZIkCfsYS5IkSYCFsSRJnZLkXUl2Jbmmp+2gJJcm+VJ7f2DbniRvT3J9ks8lObrnNae243+pvciGpHlYGEuS1C1bgONntG0CLquqI4HL2ucAzwKObG8bgHdAU0gDrwWeBBwDvHa6mJY0NwtjSZI6pKo+AXxjRvOJwLnt43OB5/e0v6c9puhTwAFJDgGeCVxaVd+oqtuAS7lvsS1phk6crm0h1m66pO9pbDvz2QOIRNJSmMNSXybqB5dUvhmYaB8fCtzUM972tm2u9vtIsoFmazMTExNMTU0NLupF2r1791jnP44Ytu64Y9b2if3gD89b2FWS1x360EGGdC9deE9GGceyKYwlSRJUVSWpAU5vM7AZYP369TU5OTmoSS/a1NQU45z/OGI4bY6NBhvX7eGsrQsr07adMjnAiO6tC+/JKOOwK4UkSd13S9tFgvZ+V9u+Azi8Z7zD2ra52iXthYWxJEnddzEwfWaJU4GLetpf3J6d4ljgjrbLxceAZyQ5sD3o7hltm6S9sCuFJEkdkuR8YBI4OMl2mrNLnAlckOR04EbgpHb0jwInANcD3wJeAlBV30jy+8C/tOO9vqpmHtAnaQYLY0mSOqSqXjjHoONmGbeAM+aYzruAdw0wNGnFsyuFJEmShIWxJEmSBFgYS5IkSYCFsSRJkgRYGEuSJEnAKjsrxVIvSbtx3R5O23SJl6OVxqzfy0pvXLeHycGEIklagdxiLEmSJGFhLEmSJAEWxpIkSRJgYSxJkiQBFsaSJEkSsIDCOMnhSS5Pcm2Szyd5edt+UJJLk3ypvT+wbU+Stye5Psnnkhw97IWQtHfmsSRJ81vIFuM9wMaqOgo4FjgjyVHAJuCyqjoSuKx9DvAs4Mj2tgF4x8CjlrRY5rEkSfOYtzCuqp1V9Zn28Z3AdcChwInAue1o5wLPbx+fCLynGp8CDkhyyKADl7Rw5rEkSfNb1AU+kqwFnghcAUxU1c520M3ARPv4UOCmnpdtb9t29rSRZAPNligmJiaYmpqadZ67d+9mamqKjev2LCbUgZrYr7kwwFwxDtv0OhgnY+hODP0aVx5vXHf3YBagDxP70Zn3ryufJeO4ry7FImm0FlwYJ9kf+CDwiqr6ZpJ7hlVVJanFzLiqNgObAdavX1+Tk5Ozjjc1NcXk5CSn9XnFq35sXLeHs7buy7ZTJscy/+l1ME7G0J0Y+jHOPD7rk3ctNeyB2bhuDyd15P3rymfJOO6rS7FIGq0FnZUiyf1pfkzPq6oPtc23TO9abe93te07gMN7Xn5Y2yZpjMxjSZL2biFnpQhwDnBdVb2lZ9DFwKnt41OBi3raX9we1X4scEfPrlpJY2AeS5I0v4V0pXgK8CJga5Kr27bXAGcCFyQ5HbgROKkd9lHgBOB64FvASwYZsKQlMY8lSZrHvIVxVX0SyByDj5tl/ALO6DMuSQNkHksrQ5LfAH4FKGArzZ/WQ4D3Aw8DrgJeVFXfTfJA4D3ATwJfB36xqraNI25pufDKd5IkLQNJDgV+HVhfVY8D9gFOBt4EvLWqHg3cBpzevuR04La2/a3teJL2wsJYkqTlY19gvyT7Ag+mOYXi04AL2+Ezz0c+fZ7yC4Hj0nsqGkn3sajzGEuSpPGoqh1J3gx8Ffg28HGarhO3V9X0yf6nzzkOPecjr6o9Se6g6W7xtd7pLvR85KPQhXNIjzqGua7TMH0NhYUYZrxdeE9GGYeFsSRJy0CSA2m2Ah8B3A78JXB8v9Nd6PnIR6EL55AedQxzXadh+hoKCzHM6yx04T0ZZRx2pZAkaXl4OvCVqrq1qr4HfIjmjDMHtF0r4N7nHL/nfOTt8IfSHIQnaQ4WxpIkLQ9fBY5N8uC2r/BxwLXA5cAL2nFmno98+jzlLwD+vj3jjKQ5WBhLkrQMVNUVNAfRfYbmVG33o+kC8SrglUmup+lDfE77knOAh7XtrwQ2jTxoaZmxj7EkSctEVb0WeO2M5huAY2YZ99+BXxhFXNJK4RZjSZIkCQtjSZIkCbAwliRJkgALY0mSJAmwMJYkSZIAC2NJkiQJsDCWJEmSAAtjSZIkCbAwliRJkgALY0mSJAmwMJYkSZIAC2NJkiQJsDCWJEmSAAtjSZIkCVhAYZzkXUl2Jbmmp+11SXYkubq9ndAz7NVJrk/yxSTPHFbgkhbOPJYkaX4L2WK8BTh+lva3VtUT2ttHAZIcBZwM/Hj7mj9Jss+ggpW0ZFswjyVJ2qt5C+Oq+gTwjQVO70Tg/VX1nar6CnA9cEwf8UkaAPNYkqT57dvHa1+a5MXAlcDGqroNOBT4VM8429u2+0iyAdgAMDExwdTU1Kwz2b17N1NTU2xct6ePUPszsR9sXLdnzhiHbXodjJMxdCeGARtZHm9cd/cAw16aif3ozPvXlc+ScdxXl2KRNFpLLYzfAfw+UO39WcAvL2YCVbUZ2Aywfv36mpycnHW8qakpJicnOW3TJUsMtX8b1+3hrK37su2UybHMf3odjJMxdCeGARppHp/1ybv6iXUgNq7bw0kdef+68lkyjvvqUiySRmtJhXFV3TL9OMmfAx9pn+4ADu8Z9bC2bUVYO4DifNuZzx5AJFL/zOOlM48laWVa0unakhzS8/Tngekj3S8GTk7ywCRHAEcCn+4vREnDYB5Ly0+SA5JcmOQLSa5L8uQkByW5NMmX2vsD23GT5O3tGWY+l+Tocccvdd28W4yTnA9MAgcn2Q68FphM8gSaXbDbgP8GUFWfT3IBcC2wBzijqsbfsVBa5cxjacU4G/jbqnpBkgcADwZeA1xWVWcm2QRsAl4FPIvmj+2RwJNouk89aTxhS8vDvIVxVb1wluZz9jL+G4E39hOUpMEyj6XlL8lDgZ8GTgOoqu8C301yIs0fX4BzgSmawvhE4D1VVcCn2q3Nh1TVzhGHLi0b/ZyVQpIkjc4RwK3Au5M8HrgKeDkw0VPs3gxMtI8PBW7qef30GWbuVRgv9Owyo9CFM4KMOoa5zro1fUashRhmvF14T0YZh4WxJEnLw77A0cDLquqKJGfTdJu4R1VVklrMRBd6dplR6MIZQUYdw1xn3Zo+I9ZCDPOsWV14T0YZx5IOvpMkSSO3HdheVVe0zy+kKZRvmT6Ytr3f1Q5f0WeYkYbBwliSpGWgqm4GbkrymLbpOJqDZC8GTm3bTgUuah9fDLy4PTvFscAd9i+W9s6uFJIkLR8vA85rz0hxA/ASmo1cFyQ5HbgROKkd96PACTSXdf9WO66kvbAwliRpmaiqq4H1sww6bpZxCzhj2DFJK4ldKSRJkiQsjCVJkiTAwliSJEkCLIwlSZIkwMJYkiRJAiyMJUmSJMDCWJIkSQIsjCVJkiTAwliSJEkCLIwlSZIkwMJYkiRJAiyMJUmSJMDCWJIkSQJg33EHsNqs3XTJol+zcd0eTut53bYznz3IkCQt0lLyeKYtx68ZQCSSpEFyi7EkSZKEhbEkSZIELKAwTvKuJLuSXNPTdlCSS5N8qb0/sG1PkrcnuT7J55IcPczgJS2MeSxJ0vwWssV4C3D8jLZNwGVVdSRwWfsc4FnAke1tA/COwYQpqU9bMI8lSdqreQvjqvoE8I0ZzScC57aPzwWe39P+nmp8CjggySEDilXSEpnHkiTNb6lnpZioqp3t45uBifbxocBNPeNtb9t2MkOSDTRbo5iYmGBqamrWGe3evZupqSk2rtuzxFD7N7EfnZr/XOtqmKbfh3EyhoEbaR5vXHf3YKLuw7hzuVdXPkvGcV9dikXSaPV9uraqqiS1hNdtBjYDrF+/viYnJ2cdb2pqisnJyXudrmzUNq7bw1lbx3dmu5nz33bK5MhjmH4fxskYhmcUeXzWJ+/qK8ZBGHcu99py/JpOfJa68pnuShzQrVhmSrIPcCWwo6qek+QI4P3Aw4CrgBdV1XeTPBB4D/CTwNeBX6yqbWMKW1o2lnpWilumd62297va9h3A4T3jHda2Seoe81hafl4OXNfz/E3AW6vq0cBtwOlt++nAbW37W9vxJM1jqYXxxcCp7eNTgYt62l/cHtV+LHBHz65aSd1iHkvLSJLDgGcD72yfB3gacGE7ysxjBaaPIbgQOK4dX9JezLtPMcn5wCRwcJLtwGuBM4ELkpwO3Aic1I7+UeAE4HrgW8BLhhCzpEUyj6UV4W3AbwMPaZ8/DLi9qqY7zk8fDwA9xwpU1Z4kd7Tjf23mRBd6rMAodKF/96hjmOu4h8UcEzHMeLvwnowyjnkL46p64RyDjptl3ALO6Dco7d0gLkfrZaVXF/O4e7buuKPvYyfM49UjyXOAXVV1VZLJQU57occKjEIX+nePOoa5vgcWc0zEMI896sJ7Mso4unEUiiRJ2punAM9LcgLwIOCHgLNpTqe4b7vVuPd4gOljBbYn2Rd4KM1BeJL2wktCS5LUcVX16qo6rKrWAicDf19VpwCXAy9oR5t5rMD0MQQvaMdf9JlnpNXGwliSpOXrVcArk1xP04f4nLb9HOBhbfsr+cGVLSXthV0pJElaRqpqCphqH98AHDPLOP8O/MJIA5NWALcYS5IkSVgYS5IkSYCFsSRJkgRYGEuSJEmAhbEkSZIEWBhLkiRJgIWxJEmSBFgYS5IkSYCFsSRJkgRYGEuSJEmAl4RetdZuumRR429ct4fTZrxm25nPHmRIkhZpsXk8my3HrxlAJJK0MrjFWJIkScLCWJIkSQIsjCVJkiTAwliSJEkCLIwlSZIkwMJYkiRJAiyMJUmSJKDP8xgn2QbcCdwN7Kmq9UkOAj4ArAW2ASdV1W39hSlpWMxjSZIag9hi/LNV9YSqWt8+3wRcVlVHApe1zyV1m3ksSVr1hnHluxOByfbxucAU8KohzEcrwGKu3DXb1ffAK/ANiXmsBfMKfKOR5HDgPcAEUMDmqjp7rj08SQKcDZwAfAs4rao+M47YpeWi38K4gI8nKeDPqmozMFFVO9vhN9Mk8H0k2QBsAJiYmGBqamrWGezevZupqSk2rtvTZ6hLN7Efq3r+c8Uw13u2GItZrrnWwyDiWKjpz+MKM5I83rju7kHHvWhdyKVpXYllEJ/pQSxHl3KrS7HMsAfYWFWfSfIQ4KoklwKn0ezhOTPJJpo9PK8CngUc2d6eBLyjvZc0h34L46dW1Y4kPwxcmuQLvQOrqtof2/tof3w3A6xfv74mJydnncHU1BSTk5OzbikclY3r9nDW1mFsXF8e858rhm2nTPY93cW8r3Oth0HEsVDTn8cVZiR5fNYn7xps1EvQhVya1pVYthy/pu/P9CC+nwcRx6B0Nc/bP6s728d3JrkOOJS59/CcCLynqgr4VJIDkhzS86dX0gx9fStX1Y72fleSDwPHALdMJ16SQ4BdA4hT0pCYx9Lyk2Qt8ETgCubew3MocFPPy7a3bfcqjBe652cUurC1ftQxzLXHZTF7lYYZbxfek1HGseTCOMka4H7tv9Y1wDOA1wMXA6cCZ7b3Fw0iUHXPIPoVarzMY23dccdY98hp8ZLsD3wQeEVVfbPpStzY2x6euSx0z88odGFr/ahjmCv/FrNXaZh7Trvwnowyjn62GE8AH24Tcl/gfVX1t0n+BbggyenAjcBJ/YcpaUjMY2kZSXJ/mqL4vKr6UNs81x6eHcDhPS8/rG2TNIclF8ZVdQPw+Fnavw4c109QkkbDPJaWj/YsE+cA11XVW3oGzbWH52LgpUneT3PQ3R32L5b2bvxHfkiSpIV4CvAiYGuSq9u219AUxLPt4fkozanarqc5XdtLRhqttAxZGEuStAxU1SeBzDH4Pnt42rNRnDHUoKQVZhBXvpMkSZKWPQtjSZIkCQtjSZIkCbAwliRJkgALY0mSJAmwMJYkSZIAC2NJkiQJ8DzGWgHWznGd+cXYduazBxCJpKXauuMOTuszl81jSf1yi7EkSZKEhbEkSZIEWBhLkiRJgIWxJEmSBFgYS5IkSYBnpZCAhZ/ZYuO6PXMeOe8R8dJ4eYYaSf1yi7EkSZKEhbEkSZIEWBhLkiRJgIWxJEmSBFgYS5IkScAQC+Mkxyf5YpLrk2wa1nwkDYc5LC1/5rG0OEMpjJPsA/wx8CzgKOCFSY4axrwkDZ45LC1/5rG0eMPaYnwMcH1V3VBV3wXeD5w4pHlJGjxzWFr+zGNpkYZVGB8K3NTzfHvbJml5MIel5c88lhZpbFe+S7IB2NA+3Z3ki3OMejDwtdFENbtfH3MM456/MSwshrxp3pc/ctDxjNtyymPoxmdoWldiMY57a/N4b7Gs5jwehS58DroQw6JyYgG/P/3oxPpgsHHMmcfDKox3AIf3PD+sbbtHVW0GNs83oSRXVtX6wYa3OOOOYdzzN4ZuxTAi8+YwLK887lIc0J1YjOO+uhRLnwaax6PQhXXfhRiMY3xxDKsrxb8ARyY5IskDgJOBi4c0L0mDZw5Ly595LC3SULYYV9WeJC8FPgbsA7yrqj4/jHlJGjxzWFr+zGNp8YbWx7iqPgp8dACT6sLunXHHMO75gzFM60IMIzHAHIburLeuxAHdicU47qtLsfRlwHk8Cl1Y912IAYxjppHEkaoaxXwkSZKkTvOS0JIkSRIdLoxHdRnLJIcnuTzJtUk+n+TlbftBSS5N8qX2/sC2PUne3sb1uSRHDzCWfZJ8NslH2udHJLmindcH2oMnSPLA9vn17fC1A5r/AUkuTPKFJNclefIo10OS32jfg2uSnJ/kQcNeB0nelWRXkmt62ha9zElObcf/UpJT+1wVK8ao8ridV2dyuSemseZ0O+2x5vWMWEae4+20zPOOmZkbc4zz/yWpJEM7E8F8cSQ5qec75X3jiCPJj7bfbZ9tP5MnDCmGbUm2Jrk6yZWzDB/6d+YC4zilnf/WJP+U5PEDDaCqOnejOUjgy8CjgAcA/wocNaR5HQIc3T5+CPD/aC6d+QfAprZ9E/Cm9vEJwN8AAY4FrhhgLK8E3gd8pH1+AXBy+/hPgf/ePv414E/bxycDHxjQ/M8FfqV9/ADggFGtB5qTzn8F2K9n2U8b9joAfho4Grimp21RywwcBNzQ3h/YPj5wFLnS5dso87idX2dyuSemseZ0O72x5fWMOMaS4+3rzfOO3WbmxizDHwJ8AvgUsH4ccQBHAp+dfp+BHx5THJt78uIoYNuQYtgGHLyX4UP/zlxgHD/V8548a9BxDOUNHsBKeTLwsZ7nrwZePaJ5XwT8HPBF4JC27RDgi+3jPwNe2DP+PeP1Od/DgMuApwEfaT94XwP2nblOaI4wfnL7eN92vPQ5/4fS/GhlRvtI1gM/uELTQe0yfQR45ijWAbCWe/9gLmqZgRcCf9bTfq/xVuttnHnczm8sudwzvbHmdDutseb1jHmOLcfbaZjnHbnNzI05xnkb8GxgiiEVxvPFQfPn6VfGvT7az9qr2sdPBv5pSHFsY+8F6VC/Mxcax4xxDwR2DHL+Xe1KMZbLWLa76p4IXAFMVNXOdtDNwMSQY3sb8NvA99vnDwNur6o9s8znnhja4Xe04/fjCOBW4N3t7pp3JlnDiNZDVe0A3gx8FdhJs0xXMdp1MG2xy+xlV2c3tvUy5lye9jbGm9Mw5rzu1bEcB/N8nN7GvXPjXtpd9IdX1SXjjAP4MeDHkvxjkk8lOX5McbwO+KUk22nOMPKyIcVRwMeTXJXmaogzjSoH5ouj1+k0W7EHpquF8cgl2R/4IPCKqvpm77Bq/pbUEOf9HGBXVV01rHkswL40uxrfUVVPBO6i2b14j2Guh7Z/34k0P+SPANYAw/oSWrBhv/cavHHmck8MXchpGHNe9+pqjoN5Pkrz5UaS+wFvATaOM47WvjTdKSZp9hj8eZIDxhDHC4EtVXUYTXeGv2jX06A9taqOpumecEaSnx7CPAYWR5KfpSmMXzXImXe1MF7QZSwHJcn9aX5Iz6uqD7XNtyQ5pB1+CLBriLE9BXhekm3A+2l2p5wNHJBk+lzTvfO5J4Z2+EOBr/cZw3Zge1Vd0T6/kOYHdVTr4enAV6rq1qr6HvAhmvUyynUwbbHLPNLP6zIy8vXSgVye1oWchvHnda8u5TiY5+Nyn9xI8t6e4Q8BHgdMteMcC1w8hAPw5osDmvy5uKq+V1VfoTlu4cgxxHE6TV98quqfgQcBBw84jum9OlTVLuDDwDEzRhlJDiwgDpL8BPBO4MSqGuT3QmcL45FdxjJJgHOA66rqLT2DLgZObR+fStNfcbr9xe3RmccCd/TsjluSqnp1VR1WVWtplvXvq+oU4HLgBXPEMB3bC9rx+9raUVU3AzcleUzbdBxwLaNbD18Fjk3y4PY9mZ7/yNZBj8Uu88eAZyQ5sN0q9oy2bbUb6eVou5DL07qQ020c487rXl3K8ZnTN89HZI7c+KWe4XdU1cFVtbYd51PA86rqPmcnGGYcrb+i2VpMkoNpulbcMIY4vkqTLyT5jzSF8a2DjCPJmiQPmX5M8/m+ZsZow/x+WHAcSX6U5o/1i6rq/w1y/kA3D75rv/tOoPl39mXgd4Y4n6fS7EL7HHB1ezuBpi/bZcCXgL8DDmrHD/DHbVxbGfBBATRJOH0E+6OATwPXA38JPLBtf1D7/Pp2+KMGNO8nAFe26+KvaDq1j2w9AL8HfIEmCf4CeOCw1wFwPk1/x+/RbB04fSnLDPxyG8v1wEvGnT9duY0qj9t5dSqXe+IaW0630x5rXs+IZeQ53k7LPO/gbUZuvJ6mAJ45ztSwcnO+ONrPwlto/sBtpT17yhjiOAr4R5oz+1wNPGMI835UO/1/BT5P+30N/Crwqz3rY6jfmQuM453Abfzge/7KQcbgle8kSZIkutuVQpIkSRopC2NJkiQJC2NJkiQJsDCWJEmSAAtjSZIkCbAwliRJkgALY0mSJAmwMJYkSZIAC2NJkiQJsDCWJEmSAAtjSZIkCbAwliRJkgALY0mSJAmwMJYkSZIAC+NVJcmfJvkf445D0uCY15I0OKmqcccgSatKkm3ABHA3sBv4W+ClVbV7ntedBvxKVT112DFK0mrkFuOOSLLvuGOQNFLPrar9gScATwRePd5wJEkWxkOU5Ogkn01yZ5K/TPKBJG9oh00m2Z7kVUluBt6d5IFJ3pbk39rb25I8sB3/4CQfSXJ7km8k+Yck92uHvSrJjnY+X0xy3BzxbJll/huT7EqyM8lLesbdL8lZSW5MckeSTybZrx32vCSfb2OZSvIfe163LclvJflckruSnJNkIsnftPH9XZIDe8Y/Nsk/tdP61ySTA38jpA6rqpuBj9EUyCTZlOTLbb5cm+Tn2/b/CPwp8OQku5Pc3rYvJq8fluSvk3wzyb8keUOST452iSWpuyyMhyTJA4APA1uAg4DzgZ+fMdqPtMMeCWwAfgc4luYH8vHAMcDvtuNuBLYDD6fZBfsaoJI8Bngp8J+q6iHAM4FtCwzzR4CHAocCpwN/3FO0vhn4SeCn2hh/G/h+kh9rl+UVbSwfBf66Xd5p/x/wc8CPAc8F/qaN9+E0n7lfb9fRocAlwBvaefwm8MEkD19g/NKyl+Qw4FnA9W3Tl4H/TJObvwe8N8khVXUd8KvAP1fV/lV1wByT3Fte/zFwVzvOqe1NktSyMB6eY4F9gbdX1feq6kPAp2eM833gtVX1nar6NnAK8Pqq2lVVt9L8KL6oHfd7wCHAI9vp/UM1HcTvBh4IHJXk/lW1raq+vMAYv9fO73tV9VGavo6PabdE/zLw8qraUVV3V9U/VdV3gF8ELqmqS6vqezQF9H40BfS0P6yqW6pqB/APwBVV9dmq+neaPwtPbMf7JeCjVfXRqvp+VV0KXAmcsMD4peXsr5LcCdwE7AJeC1BVf1lV/9bmxAeAL9H8SV6oufJ6H5o/ra+tqm9V1bXAuYNcIEla7iyMh+cRwI6699GNN80Y59a2WOx9zY09z29s2wD+D80WpY8nuSHJJoCqup5m6+3rgF1J3p/kESzM16tqT8/zbwH7AwcDD6LZcjXbct0TY1V9v12uQ3vGuaXn8bdneb5/+/iRwC+03Shub3cNP5XmD4C00j2/3cszCTyWJu9I8uIkV/fkxOOmhy3QXHn9cJo/673fQzO/kyRpVbMwHp6dwKFJ0tN2+IxxZp4S5N9oisVpP9q2UVV3VtXGqnoU8DzgldN9iavqfe1R6o9sp/mmPmP/GvDvwH+YZdi9YmyX73BgxxLmcxPwF1V1QM9tTVWduZSgpeWoqv4vTZerNyd5JPDnNN2jHtZ2l7gGmP4e6ec0QrcCe4DDetpmfidJ0qpmYTw8/0zTzeGlSfZNciLz7w49H/jdJA9PcjDwP4H3AiR5TpJHt4XoHe20v5/kMUme1h6k9+80W2S/30/g7VbgdwFvSfKIJPskeXI7jwuAZyc5Lsn9afo+fwf4pyXM6r3Ac5M8s53Hg9qDhw6b95XSyvI2mn75B9AUv7cCtAfOPa5nvFuAw2b06V+Qqrob+BDwuiQPTvJY4MX9hS1JK4uF8ZBU1XeB/0Jz8MvtNP1pP0JTRM7lDTR9bD8HbAU+07YBHAn8HU1/wX8G/qSqLqfpX3wmzVbem4EfZjCnffrNNoZ/Ab5BsxX6flX1xXZZ/rCd53NpTjv13cXOoKpuAk6kOTDvVpotyL+Fn0utMu0xBe+h+TN8Fk2O3wKsA/6xZ9S/Bz4P3Jzka0uY1UtpDsy7GfgLmj/je/tOkqRVxQt8jFCSK4A/rap3jzsWSUryJuBHqsqzU0gSbpkbqiQ/k+RH2q4UpwI/QXOFK0kauSSPTfITaRxDs0frw+OOS5K6wqutDddjaPrkrgFuAF5QVTvHG5KkVewhNN0nHkHTVeMs4KKxRiRJHWJXCkmSJAm7UkiSJElAR7pSHHzwwbV27dpZh911112sWbNmtAF1OA7oTixdiQO6E8ve4rjqqqu+VlUr9nLXe8tj6M57NEwrfRlX+vLB/Mu40vNYWu06URivXbuWK6+8ctZhU1NTTE5OjjagDscB3YmlK3FAd2LZWxxJbpx1wAqxtzyG7rxHw7TSl3GlLx/Mv4wrPY+l1c6uFJIkSRIWxpIkSRJgYSxJkiQBFsaSJEkSYGEsSZIkAR05K8VCrN10Sd/T2HbmswcQiST1z+80SeqeZVMYS1JXzFXUbly3h9MGUPBKksbDrhSSJEkSFsaSJEkSYFcKSVrVFtPXea6uIl3p6zyIfttbjl/Zl7yWtHduMZYkSZKwMJYkSZIAC2NJkiQJsI+xtCokeRfwHGBXVT2ubTsI+ACwFtgGnFRVtyUJcDZwAvAt4LSq+ky/MWzdcUffpzLrSl9WSdLKZGEsrQ5bgD8C3tPTtgm4rKrOTLKpff4q4FnAke3tScA72nt1zCAONpMk/YBdKaRVoKo+AXxjRvOJwLnt43OB5/e0v6canwIOSHLISAKVJGmMVtUW4362rmxct4fJwYUidcFEVe1sH98MTLSPDwVu6hlve9u2kxmSbAA2AExMTDA1NTX3zPZr8qgfe5v+KM21HINYxi6ba/m6/r4sxu7duzuzPJJGb1UVxpJmV1WVpJbwus3AZoD169fX5OTknOP+4XkXcdbW/r5ytp0y9/RHaa6+0hvX7el7GbtszuXbelff0x5E//FBXI57y/Fr2NvnWNLKZlcKafW6ZbqLRHu/q23fARzeM95hbZskSSuahbG0el0MnNo+PhW4qKf9xWkcC9zR0+VCkqQVa+Xu85N0jyTnA5PAwUm2A68FzgQuSHI6cCNwUjv6R2lO1XY9zenaXjLygCVJGgMLY2kVqKoXzjHouFnGLeCM4UYkSVL3zNuVIsnhSS5Pcm2Szyd5edt+UJJLk3ypvT+wbU+Stye5Psnnkhw97IWQJEmS+rWQPsZ7gI1VdRRwLHBGkqP4wcUBjgQua5/DvS8OsIHm4gCSJElSp81bGFfVzunLwVbVncB1NOc09eIAkiRJWjEW1cc4yVrgicAV9HlxgIVeGGD6ZOvjPmn+xH7dOYl9V05A35U4oDuxdCWOlWoQl0AexPlyJUkr04IL4yT7Ax8EXlFV30xyz7ClXBxgoRcGmJqaYnJyciAnbu/HxnV7OKkjJ32fXifj1pU4oDuxdCUOSZK0eAs6j3GS+9MUxedV1YfaZi8OIEmSpBVj3i3GaTYNnwNcV1Vv6Rk0fXGAM7nvxQFemuT9wJPw4gCSpHkMopuMJPVrIV0pngK8CNia5Oq27TV4cQBJkiStIPMWxlX1SSBzDPbiAJKWFbdMSpLmsqA+xpIkSdJKZ2EsSZIkYWEsSZIkARbGkiRJEmBhLEmSJAEWxpIkSRJgYSxJkiQBFsaSJEkSYGEsrXpJfiPJ55Nck+T8JA9KckSSK5Jcn+QDSR4w7jglSRo2C2NpFUtyKPDrwPqqehywD3Ay8CbgrVX1aOA24PTxRSlJ0mhYGEvaF9gvyb7Ag4GdwNOAC9vh5wLPH09okiSNzr7jDkDS+FTVjiRvBr4KfBv4OHAVcHtV7WlH2w4cOtvrk2wANgBMTEwwNTU157wm9oON6/bMOXwlWOnLuNKXD2D37t17/RxLWtksjKVVLMmBwInAEcDtwF8Cxy/09VW1GdgMsH79+pqcnJxz3D887yLO2rqyv3I2rtuzopdxpS8fwJbj17C3z7Gklc2uFNLq9nTgK1V1a1V9D/gQ8BTggLZrBcBhwI5xBShJ0qhYGEur21eBY5M8OEmA44BrgcuBF7TjnApcNKb4JEkaGQtjaRWrqitoDrL7DLCV5jthM/Aq4JVJrgceBpwztiAlSRqRld1ZTNK8quq1wGtnNN8AHDOGcCRJGhu3GEuSJElYGEuSJEmAhbEkSZIEWBhLkiRJgIWxJEmSBCygME7yriS7klzT0/a6JDuSXN3eTugZ9uok1yf5YpJnDitwSZIkaZAWssV4C7NfIvatVfWE9vZRgCRHAScDP96+5k+S7DOoYCVJkqRhmbcwrqpPAN9Y4PROBN5fVd+pqq8A1+O5UCVJkrQM9HOBj5cmeTFwJbCxqm4DDgU+1TPO9rbtPpJsADYATExMMDU1NetMdu/ezdTUFBvX7ekj1P5N7MecMY7a9DoZt67EAd2JpStxSJKkxVtqYfwO4PeBau/PAn55MROoqs00l55l/fr1NTk5Oet4U1NTTE5OctqmS5YY6mBsXLeHk+aIcdSm18m4dSUO6E4sXYlDkiQt3pLOSlFVt1TV3VX1feDP+UF3iR3A4T2jHta2SZIkSZ22pC3GSQ6pqp3t058Hps9YcTHwviRvAR4BHAl8uu8oO2LtALZabzvz2QOIRJIkSYM2b2Gc5HxgEjg4yXbgtcBkkifQdKXYBvw3gKr6fJILgGuBPcAZVXX3UCKXJEmSBmjewriqXjhL8zl7Gf+NwBv7CUqSJEkaNa98J0mSJGFhLEmSJAEWxpIkSRJgYSxJkiQBFsbSqpfkgCQXJvlCkuuSPDnJQUkuTfKl9v7AcccpSdKwWRhLOhv426p6LPB44DpgE3BZVR0JXNY+lyRpRbMwllaxJA8Ffpr2FIxV9d2quh04ETi3He1c4PnjiE+SpFFa0pXvJK0YRwC3Au9O8njgKuDlwETP1S1vBiZme3GSDcAGgImJCaampuac0cR+sHHdnsFF3kErfRlX+vIB7N69e6+fY0krm4WxtLrtCxwNvKyqrkhyNjO6TVRVJanZXlxVm4HNAOvXr6/Jyck5Z/SH513EWVtX9lfOxnV7VvQyrvTlA9hy/Br29jmWtLLZlUJa3bYD26vqivb5hTSF8i1JDgFo73eNKT5JkkbGwlhaxarqZuCmJI9pm44DrgUuBk5t204FLhpDeJIkjdTK3icmaSFeBpyX5AHADcBLaP40X5DkdOBG4KQxxidJ0khYGEurXFVdDayfZdBxIw5FkqSxsiuFJEmShIWxJEmSBFgYS5IkSYCFsSRJkgRYGEuSJEmAhbEkSZIEeLq2kVu76ZK+p7Hl+DUDiESSJEm93GIsSZIkYWEsSZIkAQsojJO8K8muJNf0tB2U5NIkX2rvD2zbk+TtSa5P8rkkRw8zeEmSJGlQFrLFeAtw/Iy2TcBlVXUkcFn7HOBZwJHtbQPwjsGEKUmSJA3XvIVxVX0C+MaM5hOBc9vH5wLP72l/TzU+BRyQ5JABxSpJkiQNzVLPSjFRVTvbxzcDE+3jQ4Gbesbb3rbtZIYkG2i2KjMxMcHU1NSsM9q9ezdTU1NsXLdniaEOxsR+jD2GadPrZNy6Egd0J5auxCFJkhav79O1VVUlqSW8bjOwGWD9+vU1OTk563hTU1NMTk5y2gBOc9aPjev2cNbWbpzdbsvxa5hrfY3S9HvTBV2JpStxSJKkxVvqWSlume4i0d7vatt3AIf3jHdY2yZJkiR12lIL44uBU9vHpwIX9bS/uD07xbHAHT1dLiRJkqTOmrdvQJLzgUng4CTbgdcCZwIXJDkduBE4qR39o8AJwPXAt4CXDCFmSZIkaeDmLYyr6oVzDDpulnELOKPfoCRJkqRR68bRZFqUrTvu6PtgxG1nPntA0WglSLIPcCWwo6qek+QI4P3Aw4CrgBdV1XfHGaMkScPmJaElAbwcuK7n+ZuAt1bVo4HbgNPHEpUkSSNkYSytckkOA54NvLN9HuBpwIXtKL0X8ZEkacWyK4WktwG/DTykff4w4Paqmr6izfSFeu5joRfqgW5dJGdYVvoyrvTlAy/SI612FsbSKpbkOcCuqroqyeRiX7/QC/UA/OF5F3XmIjnD0qULAQ3DSl8+6M4FlCSNx8r+hpM0n6cAz0tyAvAg4IeAs4EDkuzbbjX2Qj2SpFXBPsbSKlZVr66qw6pqLXAy8PdVdQpwOfCCdrTei/hIkrRiWRhLms2rgFcmuZ6mz/E5Y45HkqShsyuFJACqagqYah/fABwzzngkSRo1txhLkiRJWBhLkiRJgIWxJEmSBFgYS5IkSYCFsSRJkgR4VopVa+2mS/qexpbj1wwgEkmSpG5wi7EkSZKEhbEkSZIEWBhLkiRJgIWxJEmSBFgYS5IkSYCFsSRJkgRYGEuSJElAn+cxTrINuBO4G9hTVeuTHAR8AFgLbANOqqrb+gtTkiRJGq5BbDH+2ap6QlWtb59vAi6rqiOBy9rnkiRJUqcN48p3JwKT7eNzgSngVUOYj1aAQVyBb9uZzx5AJJIkabXrtzAu4ONJCvizqtoMTFTVznb4zcDEbC9MsgHYADAxMcHU1NSsM9i9ezdTU1NsXLenz1D7M7EfY49hWldimX5v+jGI5ZiamhpILIPQlTgkSdLi9VsYP7WqdiT5YeDSJF/oHVhV1RbN99EW0ZsB1q9fX5OTk7POYGpqisnJSU4bwJbFfmxct4eztg5jA/vidSWWLcevYa73baEG8b5uO2Xyns/JuHUljoVKcjjwHpo/sAVsrqqzPVZAkrQa9dXHuKp2tPe7gA8DxwC3JDkEoL3f1W+QkoZmD7Cxqo4CjgXOSHIUHisgSVqFlrzZMcka4H5VdWf7+BnA64GLgVOBM9v7iwYRqLpn6447xr4lX/1puz3tbB/fmeQ64FA8VkCStAr1sz9+AvhwkunpvK+q/jbJvwAXJDkduBE4qf8wJQ1bkrXAE4ErGPCxAtCdvvHDtNKXcaUvH3icgLTaLbkwrqobgMfP0v514Lh+gpI0Wkn2Bz4IvKKqvtn+4QUGc6wAwB+ed1En+sYPU1f6/w/LSl8+GMyxE5KWL698J61ySe5PUxSfV1Ufaps9VkCStOpYGEurWJpNw+cA11XVW3oGTR8rAB4rIElaJVb2PjFJ83kK8CJga5Kr27bX0Bw867ECkqRVxcJYWsWq6pNA5hjssQKSpFXFrhSSJEkSFsaSJEkSYGEsSZIkARbGkiRJEmBhLEmSJAGelUIrwNpNl7Bx3R5O23TJkqex7cxnDzAiSZK0HLnFWJIkScLCWJIkSQIsjCVJkiTAwliSJEkCLIwlSZIkwLNSSEBzZot+eWYLSZKWN7cYS5IkSVgYS5IkSYCFsSRJkgRYGEuSJEmAhbEkSZIEWBhLkiRJwBAL4yTHJ/likuuTbBrWfCQNhzksSVpthlIYJ9kH+GPgWcBRwAuTHDWMeUkaPHNYkrQaDWuL8THA9VV1Q1V9F3g/cOKQ5iVp8MxhSdKqk6oa/ESTFwDHV9WvtM9fBDypql7aM84GYEP79DHAF+eY3MHA1wYe5OJ1JQ7oTixdiQO6E8ve4nhkVT18lMEs1UJyuG1faB5Dd96jYVrpy7jSlw/mX8Zlk8eSFm9sl4Suqs3A5vnGS3JlVa0fQUjLIg7oTixdiQO6E0tX4hiVheYxrI51s9KXcaUvH6yOZZQ0t2F1pdgBHN7z/LC2TdLyYA5LkladYRXG/wIcmeSIJA8ATgYuHtK8JA2eOSxJWnWG0pWiqvYkeSnwMWAf4F1V9fklTm5Bu2lHoCtxQHdi6Uoc0J1YuhJHXwacw9NWxLqZx0pfxpW+fLA6llHSHIZy8J0kSZK03HjlO0mSJAkLY0mSJAnocGE8zsvRJnlXkl1JrulpOyjJpUm+1N4fOII4Dk9yeZJrk3w+ycvHGMuDknw6yb+2sfxe235Ekiva9+kD7YFaQ5dknySfTfKRccWRZFuSrUmuTnJl2zby92a5mPmerSRJDkhyYZIvJLkuyZPHHdOgJfmNNvevSXJ+kgeNO6Z+dOV7XlK3dLIw7sDlaLcAx89o2wRcVlVHApe1z4dtD7Cxqo4CjgXOaNfDOGL5DvC0qno88ATg+CTHAm8C3lpVjwZuA04fQSwALweu63k+rjh+tqqe0HPe03G8N8vFzPdsJTkb+NuqeizweFbYciY5FPh1YH1VPY7mgMyTxxtV37bQje95SR3SycKYMV+Otqo+AXxjRvOJwLnt43OB548gjp1V9Zn28Z00P7aHjimWqqrd7dP7t7cCngZcOMpYkhwGPBt4Z/s844hjDiN/b5aDme/ZSpLkocBPA+cAVNV3q+r2sQY1HPsC+yXZF3gw8G9jjqcvXfmel9QtXS2MDwVu6nm+vW0bp4mq2tk+vhmYGOXMk6wFnghcMa5Y2l3hVwO7gEuBLwO3V9WedpRRvU9vA34b+H77/GFjiqOAjye5qr00Moz5c9Jhb+Pe79lKcgRwK/DutqvIO5OsGXdQg1RVO4A3A18FdgJ3VNXHxxvVUJi/0irX1cK406o5x93IznOXZH/gg8Arquqb44qlqu6uqifQXAXtGOCxo5hvryTPAXZV1VWjnvcsnlpVR9N0+TkjyU/3Dhz156SrOvaeDcO+wNHAO6rqicBdrLBd8G1f2xNp/gQ8AliT5JfGG9Vwmb/S6tTVwriLl6O9JckhAO39rlHMNMn9aYri86rqQ+OMZVq7m/hy4MnAAe2uVRjN+/QU4HlJttF0sXkaTf/OUccxvRWNqtoFfJjmz8JY35uOus97luS94w1poLYD26vqivb5hTSF8krydOArVXVrVX0P+BDwU2OOaRjMX2mV62ph3MXL0V4MnNo+PhW4aNgzbPvOngNcV1VvGXMsD09yQPt4P+DnaPo8Xw68YFSxVNWrq+qwqlpL87n4+6o6ZdRxJFmT5CHTj4FnANcwhvem6+Z4z1bM1saquhm4Kclj2qbjgGvHGNIwfBU4NsmD2++l41hhBxi2zF9plevsle+SnEDTL3H6crRvHOG8zwcmgYOBW4DXAn8FXAD8KHAjcFJVzTxwY9BxPBX4B2ArP+ib+RqafsajjuUnaA5G2YfmD9UFVfX6JI+i2Qp4EPBZ4Jeq6jvDjKUnpkngN6vqOaOOo53fh9un+wLvq6o3JnkYI35vlpPe92zMoQxUkifQHFj4AOAG4CVVddtYgxqw9hSNv0hztpzPAr8yqlwfhq58z0vqls4WxpIkSdIodbUrhSRJkjRSFsaSJEkSFsaSJEkSYGEsSZIkARbGkiRJEmBhLEmSJAEWxpIkSRIA/3/CLXiFPcDW+gAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Seeing-the-distribution-graphs,-it-seems-some-values-are-really-correlated-with-each-other">Seeing the distribution graphs, it seems some values are really correlated with each other<a class="anchor-link" href="#Seeing-the-distribution-graphs,-it-seems-some-values-are-really-correlated-with-each-other">&#182;</a></h5>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-check-the-correlation-as-well.">Lets check the correlation as well.<a class="anchor-link" href="#Lets-check-the-correlation-as-well.">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[27]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Using pairplot for correlations</span>
<span class="n">sns</span><span class="o">.</span><span class="n">pairplot</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">df2</span><span class="p">,</span><span class="nb">vars</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;Total&quot;</span><span class="p">,</span><span class="s2">&quot;gross income&quot;</span><span class="p">,</span><span class="s2">&quot;cogs&quot;</span><span class="p">,</span><span class="s2">&quot;Tax 5%&quot;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[27]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;seaborn.axisgrid.PairGrid at 0x15923a0b0&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h4 id="Oh,-all-these-colums-are-extremely-correlated-with-each-other.">Oh, all these colums are extremely correlated with each other.<a class="anchor-link" href="#Oh,-all-these-colums-are-extremely-correlated-with-each-other.">&#182;</a></h4>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="A-better-way-for-checking-correlation-is-using-heatmaps">A better way for checking correlation is using heatmaps<a class="anchor-link" href="#A-better-way-for-checking-correlation-is-using-heatmaps">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[28]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">corr</span><span class="p">(),</span><span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;BuGn&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Okay,-lets-move-further,">Okay, lets move further,<a class="anchor-link" href="#Okay,-lets-move-further,">&#182;</a></h3>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-check-for-a-relationship-between-gross-income-and-customer-ratings">Lets check for a relationship between gross income and customer ratings<a class="anchor-link" href="#Lets-check-for-a-relationship-between-gross-income-and-customer-ratings">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[29]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#sns.scatterplot(df.Rating,df[&quot;gross income&quot;])</span>
<span class="n">sns</span><span class="o">.</span><span class="n">regplot</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">Rating</span><span class="p">,</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;gross income&quot;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[29]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:xlabel=&#39;Rating&#39;, ylabel=&#39;gross income&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[30]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">pairplot</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">df2</span><span class="p">,</span><span class="nb">vars</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;Rating&quot;</span><span class="p">,</span><span class="s2">&quot;gross income&quot;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[30]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;seaborn.axisgrid.PairGrid at 0x16b86cc70&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="What-can-we-say-about-branchwise-gross-income??">What can we say about branchwise gross income??<a class="anchor-link" href="#What-can-we-say-about-branchwise-gross-income??">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[31]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">df</span><span class="o">.</span><span class="n">Branch</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;gross income&quot;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[31]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:xlabel=&#39;Branch&#39;, ylabel=&#39;gross income&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="What-about-genderwise-gross-income?">What about genderwise gross income?<a class="anchor-link" href="#What-about-genderwise-gross-income?">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[32]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">df</span><span class="o">.</span><span class="n">Gender</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;gross income&quot;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[32]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:xlabel=&#39;Gender&#39;, ylabel=&#39;gross income&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Lets-see-how-gross-income-varies-over-time..">Lets see how gross income varies over time..<a class="anchor-link" href="#Lets-see-how-gross-income-varies-over-time..">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[33]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">lineplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">df2</span><span class="o">.</span><span class="n">Date</span><span class="p">,</span><span class="n">y</span><span class="o">=</span><span class="n">df2</span><span class="p">[</span><span class="s2">&quot;gross income&quot;</span><span class="p">])</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[33]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;AxesSubplot:xlabel=&#39;Date&#39;, ylabel=&#39;gross income&#39;&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="Okay,-lets-see-correlation-of-all-numeric-variables">Okay, lets see correlation of all numeric variables<a class="anchor-link" href="#Okay,-lets-see-correlation-of-all-numeric-variables">&#182;</a></h5>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h5 id="This-May-take-time-for-large-datasets">This May take time for large datasets<a class="anchor-link" href="#This-May-take-time-for-large-datasets">&#182;</a></h5>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[34]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">pairplot</span><span class="p">(</span><span class="n">df2</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[34]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>&lt;seaborn.axisgrid.PairGrid at 0x16ba4b130&gt;</pre>
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Helpful-Links">Helpful Links<a class="anchor-link" href="#Helpful-Links">&#182;</a></h3>
</div>
</div>
</div>
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<ol>
<li>More visualizations: <a href="https://www.data-to-viz.com/">https://www.data-to-viz.com/</a></li>
<li>Seaborn gallery: <a href="https://seaborn.pydata.org/examples/index.html">https://seaborn.pydata.org/examples/index.html</a></li>
<li>Pandas profiling documentation: <a href="https://pypi.org/project/pandas-profiling/">https://pypi.org/project/pandas-profiling/</a></li>
</ol>

</div>
</div>
</div>
</body>







</html>