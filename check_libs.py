import os
import sys

# Diagnostic script to find where ConversationBufferMemory is hidden
paths_to_check = [
    "langchain.memory",
    "langchain_community.memory",
    "langchain_core.memory",
    "langchain.memory.buffer",
    "langchain_community.chat_message_histories",
    "langchain.chains",
    "langchain_community.chains"
]

results = []

for p in paths_to_check:
    try:
        mod = __import__(p, fromlist=["*"])
        members = dir(mod)
        found = [m for m in members if "Memory" in m or "Chain" in m]
        results.append(f"MODULE: {p} (File: {getattr(mod, '__file__', 'N/A')})")
        results.append(f"  Members found: {found}")
        if "ConversationBufferMemory" in members:
            results.append(f"  *** FOUND ConversationBufferMemory in {p} ***")
        if "ConversationalRetrievalChain" in members:
            results.append(f"  *** FOUND ConversationalRetrievalChain in {p} ***")
    except Exception as e:
        results.append(f"ERROR checking {p}: {e}")

with open("diag_output.txt", "w") as f:
    f.write("\n".join(results))
    f.write("\n\nPYTHON PATH:\n")
    f.write("\n".join(sys.path))
