"""
クウェスト合同会社 Agentic RAG テスト
=====================================
「Claudeが自分で何を調べたか」を可視化するテストです。

【Agentic RAGとは】
通常RAG: 検索1回 → Claude に渡す → 回答
Agentic: 検索1回 → Claude が「まだ情報が足りない」と判断 → 自分でクエリを作って追加検索 → 繰り返す → 回答

このテストでは：
  1. Claude が何回追加検索したか
  2. Claude が自分で作ったクエリ（何を調べようとしたか）
  3. 最終回答
を表示します。

実行方法:
  cd D:\\qwst_git\\qwst-plus\\chatbot_marutitenant\\chatbot_backend
  python test_agentic_rag_qwest.py

結果は _agentic_result_qwest.txt にも保存されます。
"""
import os
import sys
import io
from dotenv import load_dotenv

load_dotenv()

from agentic_rag import agentic_answer

TENANT_ID = os.getenv("TEST_TENANT_ID", "b7a5fa4c-6dc1-40e2-aa48-d67db2c1e9bb")

# ─────────────────────────────────────────────────────────────
# テストケース設計方針
#
# ① 単純な質問（1件だけ調べれば十分）
#      → Claude は追加検索しないはず
# ② 複数トピックを1文に詰め込んだ質問
#      → Claude が自分でトピックを分割して複数回検索するはず ★ここがAgentic RAGの見せ場
# ③ 曖昧な質問
#      → Claude が意図を絞り込んで検索するはず
# ─────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "label": "① 単純（電話番号のみ）",
        "question": "電話番号を教えてください",
        "note": "単一トピック → Claude は追加検索しないと予想",
    },
    {
        "label": "② 複合（電話番号 + 代表者）★Agenticの見せ場",
        "question": "電話番号と代表者を両方教えてください",
        "note": "2つのトピック → Claude が「電話番号」「代表者」を別々に検索するはず",
    },
    {
        "label": "③ 複合（設立年 + 資本金 + 所在地）★3回検索を期待",
        "question": "会社の設立年と資本金と所在地を教えてください",
        "note": "3つのトピック → Claude が3回追加検索する可能性あり（上限は3回）",
    },
    {
        "label": "④ 曖昧（何ができる会社？）",
        "question": "クウェストはどんな会社ですか？何ができますか？",
        "note": "会社概要 + サービス → Claude が概要・サービスを個別に調べるかも",
    },
    {
        "label": "⑤ 単純（営業時間のみ）",
        "question": "何時まで営業していますか",
        "note": "単一トピック → 追加検索なしと予想",
    },
]


def run() -> str:
    lines = []

    def p(s: str = ""):
        lines.append(s)
        print(s)

    p("=" * 70)
    p("  Agentic RAG テスト（クウェスト合同会社）")
    p("  Claudeが自分で「何を・何回」調べたかを確認します")
    p("=" * 70)
    p()
    p("【読み方】")
    p("  追加検索0回 = 初回検索の情報だけで回答できた（通常RAGと同じ）")
    p("  追加検索1回以上 = Claudeが「まだ情報が足りない」と判断して自ら再検索した")
    p()

    for case in TEST_CASES:
        p("=" * 70)
        p(f"  {case['label']}")
        p("=" * 70)
        p(f"  質問: {case['question']}")
        p(f"  ポイント: {case['note']}")
        p()

        result = agentic_answer(case["question"], tenant_id=TENANT_ID)

        search_log = result["search_log"]
        total = result["total_iterations"]

        # ── 検索ログを表示 ──────────────────────────────────
        p("  ▼ Claude の検索プロセス")
        p(f"  {'─' * 50}")
        for entry in search_log:
            it = entry["iteration"]
            prefix = "    初回" if it == 0 else f"    第{it}回"
            label = "検索" if it == 0 else "追加検索（Claudeが自律判断）"
            p(f"  {prefix} {label}")
            p(f"         クエリ : {entry['query']}")
            if it > 0:
                p(f"         理由   : {entry['reason']}")
            p(f"         ヒット : {entry['hits']}件")
            p()

        # ── 結果サマリー ───────────────────────────────────
        if total == 0:
            summary = f"追加検索 なし（初回の情報だけで回答）"
        else:
            summary = f"追加検索 {total}回（Claudeが自律的に{total}回再検索した）"
        p(f"  ▼ 結果: {summary}")
        p()

        # ── 最終回答 ────────────────────────────────────────
        p("  ▼ Claudeの最終回答")
        p(f"  {'─' * 50}")
        for line in result["answer"].splitlines():
            p(f"  {line}")
        p()

    # ── 総括 ─────────────────────────────────────────────────
    p("=" * 70)
    p("  【Agentic RAGが効く場面まとめ】")
    p("=" * 70)
    p()
    p("  ケース②「電話番号と代表者」のように複数のトピックを")
    p("  1つの質問に詰め込んだ場合、通常RAGでは：")
    p("    → 検索結果の上位5件に全情報が揃わないことがある")
    p("    → Claudeが「資料に情報がありません」と答える可能性がある")
    p()
    p("  Agentic RAGでは：")
    p("    → Claudeが自分でトピックを分割して追加検索する")
    p("    → 各トピックの情報を集めて完全な回答を作る")
    p()
    p("  ※ 今回のDBは7件のみなので差が小さいですが、")
    p("     数百件の実際の運用データでは差が顕著になります。")
    p()

    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    result_text = run()
    out_path = os.path.join(os.path.dirname(__file__), "_agentic_result_qwest.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"\n→ 結果を保存しました: {out_path}")
