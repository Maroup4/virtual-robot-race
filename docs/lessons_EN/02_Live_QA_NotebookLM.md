# 2. Live Q&A with NotebookLM

## Introduction
As you develop in aira, you will encounter errors and moments when you lose track of your code's structure.
When that happens, your powerful supporter is the **NotebookLM Live Q&A System**.

This tool acts as an "AI mentor" that fully understands aira's specifications and actual source code.

> **▶ Watch this lesson first**: [Lesson 02 NotebookLM Guide (YouTube)](https://youtu.be/kwLAPdCD-jU?t=150)

> **❓ "Can't I just use ChatGPT or Gemini?"**
> General AI chatbots answer from broad training data — but they have not seen aira's source code or specifications. NotebookLM runs with aira's **repository and documentation loaded as its source**, so it can accurately answer project-specific questions like "where is this function defined?" or "which part of aira's code triggers this error?"

## 2.1 How to Access the Live Q&A
Use the link below to open the dedicated chat window for this training.

[Ask a question on NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e)

Login: Sign in with your own Google account.

Interface: Only a chat input field is displayed. You cannot access the source code or notes directly, but you can draw out the information you need through conversation.

**Try it now!** Copy and send the prompt below to get a feel for the AI mentor.

```
What are the key points to winning this race?
```

See how NotebookLM responds and get a sense of how it works.

> **❓ "Can other participants or organizers see my questions and algorithms?"**
> **No.** Chat history is tied to your Google account as a **private session**. Even though everyone uses the same link, no other participant or organizer can view your conversations. That said, anything you paste into the chat is subject to Google's terms of service and privacy policy. If you are concerned about sharing sensitive algorithm code, paste only the relevant excerpt rather than your full implementation.

## 2.2 What You Can Do with NotebookLM
You can ask this AI mentor questions such as:

- Get help resolving errors
  - "I got this error when I ran `python main.py`: [error]. How do I fix it?"

- Ask for code explanations
  - "Which part of `main.py` handles the communication with Unity (WebSocket)?"

- Confirm architecture
  - "How does the decision-making logic switch between Rule-Based mode and AI mode?"

- Check specifications
  - "What happens to the robot when SOC (battery level) reaches zero?"

> **❓ "I already have an AI coding assistant in VS Code — why open NotebookLM in a browser too?"**
> They serve different roles:
> - **AI coding assistant (Gemini Code Assist, Claude Code, Codex, etc., in VS Code)**: Best at writing, fixing, and completing code while looking at your open files. Use it for "fix this error" or "implement this function."
> - **NotebookLM (in browser)**: Acts as a dictionary of aira's official documentation and initial repository state. Best for "how does this mechanism work?" or "which file should I modify?"
>
> Think of them as "the tool that writes code" and "the aira reference manual" — using both in parallel is the most efficient workflow.

## 2.3 Tips for Getting Good Answers
To get more accurate advice from the AI, include the following information in your question:

- **What** you were trying to do: (e.g., "I was trying to run manual mode...")
- **What** error occurred: (e.g., paste the error text shown in the terminal)
- **How far** you have gotten: (e.g., "I have completed the setup steps in lesson 00")
- **Your modified code**: Paste the relevant section you changed (see below)

Bad example: "It's not working. Please help."
Good example: "I ran `main.py`, but the Unity screen stays black. The terminal shows `ConnectionRefusedError`. What should I check?"

> **❓ "Does NotebookLM know about the code I've modified locally?"**
> **No.** NotebookLM only knows the **official repository in its initial state**. Your local changes to `rule_based_algorithms/` or any parameters you just updated are invisible to it. When asking about your own code, **copy and paste the relevant section into the chat** — something like: "Please look at this code: [paste here]. Why does it behave this way?"

> **❓ "What if the AI gives me wrong information — how do I catch it?"**
> NotebookLM is more grounded than general AI, but it can still suggest variable names or functions that don't exist. **How to verify**: When it tells you "this function is in this file," use `Ctrl+Shift+F` in VS Code to search the entire project for it. If you can't find it, treat the answer with suspicion and follow up with: "I couldn't find that. Could you double-check?"

## 2.4 Integrating It into Your Workflow
During lessons, keep NotebookLM open in a browser tab while you write code in VS Code.
It lets you instantly resolve those "I don't want to interrupt the instructor, but I just need a quick check" moments — dramatically improving your learning efficiency.

---

⬅️ [Previous lesson: 01_Foundation.md (Foundation)](01_Foundation.md) ｜ ➡️ [Next lesson: 03_Manual_Control.md (Manual Control)](03_Manual_Control.md)
