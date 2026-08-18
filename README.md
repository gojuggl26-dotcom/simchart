# simchart — 段階構築式の金融マイクロ構造シミュレータ

S0〜S13 の 13 段階で作る市場マイクロ構造シミュレータ。**このリポジトリの現状は
S0 (骨格層) まで。**

S0 の価格過程は幾何ブラウン運動だけで、価格モデルとしては意図的に非現実的である。
尖度 3・|r| 自己相関ゼロ・単一フラクタル・分散比 1 が S0 の正解であり、それらしく
見せるチューニングはしていない。S0 の価値はフラグ設計・RNG 設計・検証スイート・
結果永続化にあり、この足場が S1〜S13 全部を支える。

---

## 使い方

```bash
uv sync                                  # 依存の導入 (Python 3.12+)

uv run python -m simchart.cli run --config configs/s0.yaml --stage S0
uv run python -m simchart.cli validate --stage S0          # 保存済み結果の再判定
uv run python -m simchart.cli compare --stages S0 S1       # 段階間の指標差分
uv run pytest                                              # テスト
```

`run` の主なオプション: `--seed` / `--n-days` / `--steps-per-day` /
`--results-dir` / `--no-plots`。終了コードは critical ゲート全合格で 0、
不合格があれば 1、未実装フラグに触れた場合は 3。

出力は `results/<stage>/metrics.json` と `results/<stage>/plots/*.png`。

---

## 層構造 (最終系)

```
L0 カレンダー層   φ(t) 日内U字・寄引・オーバーナイト
       ↓
L1 潜在活動度層   λ(t) = φ_λ(t)·μ·Z_t + 多変量Hawkes      ← χ₁, χ₃ 注入点
       ↓
L2 情報価格層     log v_t = ラフ(H≈0.1) + MSM + 緩慢OU + レバレッジ  ← χ₂ 注入点
                  p*(t) = マルチンゲール + Hawkesジャンプ
       ↓ κ（結合強度）
L3 板層           メタオーダー分割 → 6次元Hawkes注文流 →
                  queue-reactive板 → uncertainty zones で離散化
       ↓
   RV_t ──フィードバック──→ L1(n_t), L3(取消率/配置距離)
```

観測価格は板のミッドであり、p\*(t) は外生生成される潜在情報価格。注文流の一部が
p\* 方向にバイアスを持つ (ハイブリッド方式)。**S0 では L0 / L1 / L3 はスタブ**で、
observed price = p\*(t) をそのまま返す。

```
simchart/
  config.py            全段階のフラグ / YAML・JSON ロード / 未実装フラグの拒否
  rng.py               名前ベースの層別 RNG ストリーム
  types.py             PriceProcess, EventLog, BookSnapshot, Observation,
                       BarSeries, StageResult, 各層の Protocol
  layers/
    l0_calendar.py     STUB: phi(t) -> 1.0、等間隔セッション
    l1_activity.py     STUB: 定数強度、イベント生成は拒否
    l2_price.py        GBM 実装 + S1〜S5 の拡張フック
    l3_book.py         STUB: observed = p*
  pipeline.py          層の組み立て / 駆動方式の選択 / 決定性・RNG 安定性の検査
  validation/
    base.py            ok / not_applicable / error の共通表現
    tails.py           Hill 推定量, QQ, 基本モーメント
    memory.py          ACF, Ljung-Box, GPH, local Whittle, 冪則当てはめ
    scaling.py         分散比, スケール別尖度, zeta_q, signature plot, ADF
    micro.py           符号ACF, propagator, 平方根則, 分岐比再推定
    cross.py           Hayashi-Yoshida
    suite.py           run_all() -> dict
    gates.py           段階別ゲート定義と判定
  report.py            metrics.json の書き出し・読み直し・プロット・段階間比較
  cli.py               run / validate / compare
```

---

## 設計上の決定 (後段全部を規定するので先に読むこと)

### 1. RNG は名前ハッシュ方式 (`rng.py`)

子シードを `sha256(f"{master}:{stream_name}")` から導出する。`SeedSequence.spawn()`
は**呼び出し順に子シードを配るので使えない** — 途中に新しい spawn を挟むと以降が
全部ずれる。名前方式なら、S3 でジャンプ用ストリームを足しても拡散の経路は S2 と
ビット単位で同一のままになり、「新機能の効果」と「乱数がずれただけ」を区別できる。

- L2 と L3 のストリームは接頭辞で分離してあり、共有しない。共有すると S10 で L3 を
  変えただけで L2 の価格経路が動き、結合前後の比較が成立しなくなる。
- 未宣言のストリーム名は既定で拒否する (`strict=True`)。打ち間違いは「有効に見える
  別の系列」を静かに返すという最悪の壊れ方をするため。
- 後段で使う予定の名前は `RESERVED_STREAM_NAMES` に先に確保してある。

### 2. `PriceProcess` は配列ではなく補間可能なオブジェクト (`types.py`)

S10 で L3 が不規則なイベント時刻に p\* を参照する。`at(t)` は線形補間で、
**決定論的かつ冪等**である。問い合わせのたびに乱数を引くブラウン橋にすると
問い合わせ順序が価格経路を変えてしまい、「L3 を変えても L2 は不変」という保証が
崩れる。代償としてグリッド間隔より短いスケールで分散が過小になるので、必要に
なったら (a) L2 のグリッドを細かくする、(b) `interpolation="brownian_bridge"` を
実装して問い合わせ時刻を事前に確定させてから一括で橋を張る、の順で対処する。
`l2.bridge` ストリームは予約済み。

S3 でジャンプを入れるときは、線形補間がジャンプをなますので**ジャンプ時刻を必ず
グリッド点に載せる**こと。

### 3. セッション境界をまたぐ差分をリターンにしない

`Observation.to_bars()` は `(セッション, バー)` の 2 次元で返す。S0 にオーバーナイトは
無いので実質差は出ないが、S4 でギャップを入れた瞬間に「日をまたぐ差分」が巨大な
外れ値として ACF を汚す。最初からその構造で測る。

例外は |リターン| の長期記憶で、こちらは連結した 1 次元系列で測る。ボラティリティ
過程は日をまたいで続くものなので、日をまたぐラグを見られないと「今日のボラが明日の
ボラを予測するか」が測れない。連結で偽の値は生じない (各 |r| はセッション内で完結)。

### 4. ゲートは閾値を緩めず、測り方を選ぶ

指示書の閾値をそのまま使うと、推定量の精度が足りずに偶然でも落ちる箇所があった。
対処は閾値を緩めることではなく、閾値が意味を持つ精度で測ることにした。

| 指標 | 素朴な測り方 | 標準誤差 | ゲート幅 | 採った測り方 |
|---|---|---|---|---|
| 尖度 (スケール別) | 全スケールを判定 | 1 日スケールで 0.22 | ±0.4 | 標本数 10,000 未満のスケールは**記録のみ** |
| GPH の d | m = N^0.5 | 0.030 | ±0.05 | m = N^0.65 (標準誤差 0.011) |
| Ljung-Box | 5 ラグの最小 p 値 | — | p>0.01 | ラグ 20 単独で判定 (多重比較を避ける) |

基準リターンの粒度は 1 分 (`validation.primary_bar_sec`)。既定設定で約 195,000 本
取れ、尖度の標準誤差が 0.011 になるので `[2.7, 3.3]` が本物の検定になる。

### 5. 未実装フラグは黙って無視しない

`Config.__post_init__` が `NotImplementedError` を送出し、メッセージに実装段階と
追加先ファイルを含める。各層のビルダーでも二重にチェックしてある (Config を
経由しない直接構築で静かにスタブが使われるのを防ぐため)。

### 6. 指示書との差分

- `Config` に **`p0` (初期価格水準、既定 100.0)** を足した。S9 の tick size で
  価格水準が必要になり、そのとき追加すると設定ファイルの互換性が壊れるため。
  リターン系の統計には一切影響しない。
- 検証スイートの推定器設定を `Config.validation` (`ValidationConfig`) に分離した。
  モデル設定ではなく「測定器の設定」であり、段階をまたいで固定したいため。
- `requires-python` は 3.12 以上 (numpy 2.5 系の下限)。
- ゲートを 2 つ足した: `acf_abs_r_lag1` (指示書 §7 で「あり」とされているもの) と
  `rng_streams_distinct` (名前ハッシュの衝突検出)。

---

## S0 のゲート

| ゲート | 指標 | 条件 |
|---|---|---|
| pipeline_runs | `runtime.pipeline.completed` | 例外なく完走 |
| determinism | `runtime.determinism.bitwise_identical` | 同一シード 2 回でビット単位同一 |
| rng_stability | `runtime.rng_stability.unchanged` | ダミーストリーム追加後も既存が不変 |
| rng_streams_distinct | `runtime.rng_stability.streams_distinct` | 宣言済みストリームが相互に別系列 |
| acf_r_lag1 | `memory.acf_r.lag1_z` | \|ρ(1)\| < 2/√N |
| acf_abs_r_lag1 | `memory.acf_abs_r.lag1_z` | \|ρ(1)\| < 2/√N |
| ljung_box | `memory.ljung_box_r.pvalue_primary` | p > 0.01 (ラグ 20) |
| gph_d | `memory.gph_abs_r.d` | \|d\| < 0.05 |
| variance_ratio | `scaling.variance_ratio.max_abs_dev` | 全 q で 0.90〜1.10 |
| kurtosis | `tails.moments.kurtosis` | 2.7〜3.3 |
| kurtosis_flat | `scaling.kurtosis_by_scale.max_abs_dev_from_3_gated` | 全ゲート対象スケールで 2.6〜3.4 |
| zeta_q_linear | `scaling.zeta_q.r2` | R² > 0.99 |
| signature_plot_flat | `scaling.signature_plot.max_rel_dev` | 中央値からの最大相対乖離 < 0.10 |
| adf | `scaling.adf.combined_ok` | log P で棄却せず・r で棄却 |
| validation_callable | `runtime.validation.all_callable` | 全検証関数が例外なく呼べる |
| artifacts_written | `runtime.artifacts.metrics_json_ok` | metrics.json が存在し全項目を含む |

`results/<stage>/metrics.json` は単なるログではなく**回帰テストの基準**である。
後段で異常が出たときに「どの段階までは正常だったか」を遡るために段階ごとに残し、
git に追跡させる (プロットは再生成できるので追跡しない)。

---

## S1 で何をどのファイルのどこに追加するか

S1 の目的は**ボラティリティ・クラスタリングとファットテールを内生的に出すこと**。
MSM (マルコフ・スイッチング・マルチフラクタル) と緩慢 OU を対数ボラに足す。

### 1. `simchart/config.py`

- `IMPLEMENTED_STAGES` に `"S1"` を追加する。
- `UNIMPLEMENTED_FLAGS` から `enable_msm` と `enable_slow_ou` の 2 行を削除する。
- MSM のパラメータを `Config` に追加する
  (`msm_k_components: int = 8`, `msm_m0: float = 1.4`, `msm_gamma_1: float`,
  `msm_b: float`, `ou_kappa: float`, `ou_sigma: float`)。
  **フラグが False のときにこれらを変えても何も起きない = 暗黙 no-op になる**ので、
  「フラグが False なのにパラメータが既定値から動いていたら `ValueError`」を
  `_check_basic` に足すこと。

### 2. `simchart/rng.py`

- MSM の状態遷移は既存の `l2.vol_msm`、緩慢 OU は予約済みの `l2.vol_ou` を使う。
  `RESERVED_STREAM_NAMES` から `l2.vol_ou` を `STREAM_NAMES` へ移すだけでよい。
- **`l2.diffusion` の消費のしかたを変えないこと。** 変えると S0 との差分が
  「MSM の効果」ではなく「乱数のずれ」になり、`compare --stages S0 S1` が
  読めなくなる。`GBMPriceLayer.simulate` は今も `standard_normal(n-1)` を
  一度だけ引いている。この呼び方を維持する。

### 3. `simchart/layers/l2_price.py` — 中心的な変更

`GBMPriceLayer._log_vol_path(t)` に成分を**加法で**足す。

```python
def _log_vol_path(self, t):
    base = np.full(t.shape[0], math.log(self._config.sigma_bar))
    if self._config.enable_msm:
        base += self._msm_component(t)      # 新規: log(M_1 M_2 ... M_k)
    if self._config.enable_slow_ou:
        base += self._slow_ou_component(t)  # 新規: OU 過程 X_t
    return base
```

加法で設計する理由は、対数ボラの分散分解で各成分の寄与を切り分けられるようにする
ため。乗法で混ぜると S1 と S2 の効果が分離できなくなる。

`simulate()` 本体は**触らなくてよい**。ボラが定数でなくなると自動的に
非スカラー経路 (`uniform and np.all(sigma_left == sigma_left[0])` が False) に
入るようになっている。区間左端のボラを使う規約もそのまま維持すること
(右端や区間平均を使うと S3 でレバレッジを入れたときに未来のボラが当該区間の
リターンへ漏れる)。

### 4. `simchart/validation/gates.py`

`STAGE_GATES` に `"S1"` を追加する。S0 のゲートのうち**反転するもの**と
**維持するもの**を明示的に分けること。

| S0 のゲート | S1 での扱い |
|---|---|
| `acf_r_lag1` | **維持** (リターン自体は無相関のまま) |
| `variance_ratio` | **維持** (拡散性は保たれる) |
| `adf` | **維持** |
| `determinism` / `rng_stability` / `artifacts_written` など | **維持** |
| `acf_abs_r_lag1` | **反転** — \|r\| に正の自己相関が出ること |
| `gph_d` | **反転** — d > 0 (ただし MSM は真の長期記憶ではないので過大に出る) |
| `kurtosis` | **反転** — 3 より大きいこと |
| `kurtosis_flat` | **反転** — 集計で 3 に**近づく**こと (集計正規性)。単調減少を確認する |
| `zeta_q_linear` | **反転** — R² が下がり、ζ_q が上に凸になること |
| `signature_plot_flat` | **維持** (ノイズは S9 から) |

集計正規性の検査は `scaling.kurtosis_by_scale` の `table` からスケール順の単調性を
見る関数を `scaling.py` に足すのがよい (`kurtosis_aggregation_slope` など)。

### 5. `configs/s1.yaml`

`configs/s0.yaml` をコピーし、`stage: S1`、`enable_msm: true`、
`enable_slow_ou: true` と MSM パラメータを書く。**`validation:` セクションは
一切変えないこと。** 測定器の設定が変わると S0 との比較が成立しない。

### 6. 検証

```bash
uv run python -m simchart.cli run --config configs/s1.yaml --stage S1
uv run python -m simchart.cli compare --stages S0 S1 --only-changed
```

`compare` で確認すべきこと:

- `runtime.rng_fingerprint.*` が S0 と**完全一致**していること (乱数の割り当てが
  ずれていない証拠)。
- `memory.acf_r.lag1_z` が S0 と同程度に小さいままであること。
- `tails.moments.kurtosis` と `memory.gph_abs_r.d` が上がっていること。
- `scaling.signature_plot.max_rel_dev` が S0 と同程度であること。

---

## S2 以降の追加先

| 段階 | 内容 | 主な追加先 |
|---|---|---|
| S2 | ラフ・ボラ (H≈0.1) | `layers/l2_price.py::_log_vol_path`, ストリーム `l2.vol_rough` |
| S3 | Hawkes ジャンプ / レバレッジ | `layers/l2_price.py::_jump_component`, `_leverage_innovation`, ストリーム `l2.jump_time` / `l2.jump_size` / `l2.leverage` |
| S4 | 日内季節性 / オーバーナイト | `layers/l0_calendar.py` 全体、`simulation_grid()` に境界 2 点を置く |
| S5 | カオス的ボラ χ₂ | `layers/l2_price.py::_log_vol_path` |
| S6 | 板層の導入 | `layers/l3_book.py`, `pipeline.py::select_driver` に `EventDriver` を追加 |
| S7 | 多変量 Hawkes 注文流 | `layers/l1_activity.py::event_times`, ストリーム `l1.hawkes` |
| S8 | メタオーダー分割 | `layers/l3_book.py`, ストリーム `l3.metaorder`。制約 β=(1−γ)/2 を `validation/micro.py::impact_consistency` で常時監視する |
| S9 | queue-reactive / uncertainty zones | `layers/l3_book.py`, ストリーム `l3.queue` / `l3.uncertainty`, `Config` に `tick_size` を追加 |
| S10 | p\* と注文流の結合 (κ) | `layers/l3_book.py`。`PriceProcess.at()` の分散過小が効くなら `interpolation` を拡張 |
| S11 | RV フィードバック | `pipeline.py` に反復駆動を追加 |
| S12 | カオス χ₁ / χ₃ | `layers/l1_activity.py`, ストリーム `l1.chaos` |
| S13 | 多資産 | `pipeline.py` を資産ループ化、ストリーム `cross.factor`、`validation/cross.py` が有効化 |

**共通の作法**: 段階を進めるときは (1) `IMPLEMENTED_STAGES` に追加、
(2) `UNIMPLEMENTED_FLAGS` から該当行を削除、(3) `STAGE_GATES` にその段階の
ゲートを定義、(4) `configs/s<N>.yaml` を作成、(5) `compare --stages S<N-1> S<N>`
で「変わるべきものだけが変わった」ことを確認、の 5 点を必ず通す。

---

## S0 で意図的にやっていないこと

- t 分布その他ファットテール革新項。テールは S1〜S3 でボラ過程とジャンプから
  内生的に出す。外生的に入れると**時間集計で尖度が下がる性質 (集計正規性) が
  永久に再現できなくなる。**
- リターンへの MA(1) や人工的な自己相関。短期の負の自己相関は S9 の
  uncertainty zones から出す。
- S0 を「それらしく」見せるチューニング。尖度 3・|r| 自己相関ゼロ・単一フラクタル
  が正解。
