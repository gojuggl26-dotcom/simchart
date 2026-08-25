"""名前ベースの層別 RNG ストリーム管理。

要件 (指示書 §5)
----------------
1. 同一シードでビット単位同一の結果になること。
2. **後段で新しいストリームを追加しても、既存ストリームの系列が変わらないこと。**

2 が効いてくるのは、たとえば S3 でジャンプ成分を足したときに「拡散の経路は S2 と
完全に同一のまま、ジャンプだけが乗った」と言えるかどうか、という場面である。これが
崩れると段階間の差分が「新機能の効果」なのか「乱数がずれただけ」なのか区別できず、
段階構築という方法そのものが無意味になる。

``np.random.SeedSequence.spawn()`` は**呼び出し順に子シードを配るため要件 2 を
満たさない** (途中に新しい spawn を挟むと以降が全部ずれる)。よってストリーム名の
ハッシュから子シードを決定論的に導出する。名前が同じなら、他に何本ストリームが
あろうと、どんな順序で取得しようと、常に同じ系列になる。

L2 と L3 のストリームは絶対に共有しない
--------------------------------------
共有すると S10 (kappa による p* と注文流の結合) で L3 側を変えただけで L2 の価格
経路まで変わり、結合前後の比較が成立しなくなる。名前空間の接頭辞 (``l2.`` / ``l3.``)
で分離し、層は自分の接頭辞のストリームしか取らない。
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "RNGRegistry",
    "AssetStreamView",
    "UnknownStreamError",
    "derive_seed",
    "asset_stream_name",
    "asset_stream_names",
    "STREAM_NAMES",
    "RESERVED_STREAM_NAMES",
    "KNOWN_STREAMS",
]

#: 指示書 §5 で宣言されたストリーム名。S0 では大半が未使用だが、名前は先に確定させる。
#: (名前ハッシュ方式なので、使っていないストリームが存在しても他に影響しない)
STREAM_NAMES: tuple[str, ...] = (
    "l2.diffusion",
    "l2.vol_msm",
    "l2.vol_rough",
    "l2.jump_time",
    "l2.jump_size",
    "l1.hawkes",
    "l3.order_type",
    "l3.order_size",
    "l3.order_price",
    "l3.cancel",
    "l3.metaorder",
    # --- S1 で追加 (追加しても既存ストリームの系列は変わらない: 名前ハッシュ方式) ---
    "l2.vol_slow",  # 緩慢 OU (S1)
    "validation.ensemble",  # 検証スイートのアンサンブル断面 (S1)。生成系とは独立
    # --- S3 で追加 ---
    "l2.leverage",  # bridge のセル集計直交成分 w (S3)
    "l2.leverage_slow",  # OU 駆動の直交成分 w2 (S3)
    "l2.leverage_mid",  # 中速レバレッジ成分の x0 と直交駆動 (S3)
    # --- S4 で追加 ---
    "l0.overnight",  # オーバーナイト・ギャップの拡散とジャンプ (S4)
    # --- S13 で追加: 多資産の共通因子 (資産に依存しない市場全体の乱数) ---
    # ★共通ストリームは n_assets にも資産の追加にも依存しない (§8.2 の不変性の
    # 前提)。資産固有ストリームは下の asset_stream_name() で導出する。
    "cross.vol_msm",  # 共通 MSM (遅い側 k_c 成分)
    "cross.vol_slow",  # 共通 緩慢 OU (x0 と駆動)
    "cross.leverage_slow",  # 共通 OU 駆動の直交成分 (レバレッジ有効時)
    "cross.jump_time",  # 共通 (システマティック) ジャンプの時刻
    "cross.jump_size",  # 共通ジャンプの符号と大きさ
)

#: 後段で必要になることが設計上ほぼ確実なストリーム。先に名前だけ確保しておく。
#: 追加しても既存ストリームには一切影響しないが、名前を先に決めておくと
#: 「後で似た名前を作ってしまい系列が変わる」事故を防げる。
RESERVED_STREAM_NAMES: tuple[str, ...] = (
    # S10: PriceProcess.at() をブラウン橋で補間する方式に切り替える場合に使う。
    # S0 の既定は決定論的な線形補間なので未使用 (README §補間の設計判断 を参照)。
    "l2.bridge",
    "l0.calendar",  # S4+: 日次のカレンダー撹乱 (祝日・半日立会など)
    "l1.chaos",  # S12: chi_1 / chi_3 の初期条件
    "l3.queue",  # S9: queue-reactive 板
    "l3.uncertainty",  # S9: uncertainty zones
    "l3.latency",  # 予備: 遅延
    "cross.factor",  # S13: 多資産の共通因子
)

#: strict モードで取得を許すストリーム名の全体。
KNOWN_STREAMS: frozenset[str] = frozenset(STREAM_NAMES + RESERVED_STREAM_NAMES)


class UnknownStreamError(KeyError):
    """未宣言のストリーム名が要求された。"""


def derive_seed(master_seed: int, name: str) -> int:
    """マスターシードとストリーム名から子シードを決定論的に導出する。

    ``sha256(f"{master}:{name}")`` の先頭 8 バイトを big-endian の符号なし整数と
    して解釈する。純粋関数であり、呼び出し順序にも他ストリームの有無にも依存しない。
    """
    digest = hashlib.sha256(f"{master_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class RNGRegistry:
    """ストリーム名 -> :class:`numpy.random.Generator` の対応を管理する。

    Parameters
    ----------
    master_seed:
        全体のシード。
    strict:
        ``True`` (既定) のとき、:data:`KNOWN_STREAMS` にない名前を要求すると
        :class:`UnknownStreamError` を送出する。ストリーム名の打ち間違いは
        「有効に見える別の系列」を静かに返すという最悪の壊れ方をするため、
        既定で弾く。
    extra_streams:
        この実行に限り追加で許可する名前。新段階の実装中や試験用に使う。

    Notes
    -----
    同じ名前で 2 回 ``get`` すると**同一の Generator オブジェクト**が返る
    (キャッシュされる)。したがって同じストリームを 2 か所から引くと、消費順序が
    結果に効く。1 ストリームは 1 用途に割り当てること。
    """

    def __init__(
        self,
        master_seed: int,
        *,
        strict: bool = True,
        extra_streams: Iterable[str] = (),
    ) -> None:
        self._master = int(master_seed)
        self._strict = bool(strict)
        self._allowed: frozenset[str] = KNOWN_STREAMS | frozenset(extra_streams)
        self._cache: dict[str, np.random.Generator] = {}
        self._order: list[str] = []

    # ------------------------------------------------------------------
    @property
    def master_seed(self) -> int:
        return self._master

    def get(self, name: str) -> np.random.Generator:
        """ストリーム ``name`` の Generator を返す (初回に生成しキャッシュ)。"""
        if name not in self._cache:
            if self._strict and name not in self._allowed:
                raise UnknownStreamError(
                    f"未宣言の RNG ストリーム {name!r} です。"
                    f" simchart/rng.py の STREAM_NAMES / RESERVED_STREAM_NAMES に"
                    f" 追加してください (追加しても既存ストリームの系列は変わりません)。"
                )
            self._cache[name] = np.random.default_rng(derive_seed(self._master, name))
            self._order.append(name)
        return self._cache[name]

    def child_seed(self, name: str) -> int:
        """``name`` に対応する子シードを返す (監査・ログ用。状態を消費しない)。"""
        return derive_seed(self._master, name)

    def fingerprint(self, names: Sequence[str] | None = None) -> dict[str, int]:
        """宣言済みストリームの子シード一覧。metrics.json に載せて再現性を追跡する。"""
        target = tuple(names) if names is not None else STREAM_NAMES
        return {name: self.child_seed(name) for name in target}

    def used_streams(self) -> tuple[str, ...]:
        """この実行で実際に取得されたストリーム名 (取得順)。"""
        return tuple(self._order)

    def reset(self) -> None:
        """全ストリームを初期状態へ戻す。同一シードでの再実行に使う。"""
        self._cache.clear()
        self._order.clear()

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return (
            f"RNGRegistry(master_seed={self._master}, strict={self._strict}, "
            f"used={len(self._order)})"
        )


# ---------------------------------------------------------------------------
# S13: 資産別の名前空間
# ---------------------------------------------------------------------------
def asset_stream_name(name: str, asset_index: int) -> str:
    """資産 ``asset_index`` のストリーム名を返す。

    **資産 0 はレガシー名 (接尾辞なし) をそのまま使う。** これが §8.3 の
    「n_assets=1 は S12 と厳密一致」を成立させる: N=1 では因子パラメータが
    全て退化 (共有 0) し、資産 0 の固有ストリーム = S12 のストリームになるので、
    消費列がビット単位で一致する。資産 1 以降は ``.asset{i}`` を付ける。
    名前ハッシュ方式なので、資産を何本足しても既存資産の系列は変わらない
    (§8.2 — 逐次 spawn を使わないことがここでも効く)。

    ``cross.*`` (共通因子) は資産に属さないので接尾辞を付けてはならない。
    """
    if name.startswith("cross."):
        raise ValueError(f"共通ストリーム {name!r} に資産接尾辞は付けられません")
    if asset_index < 0:
        raise ValueError("asset_index は 0 以上である必要があります")
    return name if asset_index == 0 else f"{name}.asset{asset_index}"


def asset_stream_names(n_assets: int) -> tuple[str, ...]:
    """資産 1..n-1 の全派生ストリーム名 (RNGRegistry の extra_streams 用)。"""
    per_asset = tuple(
        n for n in STREAM_NAMES + RESERVED_STREAM_NAMES if not n.startswith("cross.")
    )
    return tuple(
        asset_stream_name(name, i)
        for i in range(1, max(n_assets, 1))
        for name in per_asset
    )


class AssetStreamView:
    """資産 1 本分の視界に絞った RNGRegistry のビュー。

    層のコード (l2_price / l3_book など) は ``rng.get("l2.diffusion")`` のように
    レガシー名で引く。このビューを渡すと名前が資産別に写像されるので、
    **層のコードは 1 文字も変えずに**多資産化できる。共通因子 (``cross.*``) は
    ビューを通さず素の registry から引くこと (資産に属さないため)。
    """

    def __init__(self, registry: RNGRegistry, asset_index: int) -> None:
        self._registry = registry
        self._asset_index = int(asset_index)

    @property
    def master_seed(self) -> int:
        return self._registry.master_seed

    @property
    def asset_index(self) -> int:
        return self._asset_index

    def _map(self, name: str) -> str:
        return asset_stream_name(name, self._asset_index)

    def get(self, name: str) -> np.random.Generator:
        return self._registry.get(self._map(name))

    def child_seed(self, name: str) -> int:
        return self._registry.child_seed(self._map(name))

    def fingerprint(self, names: Sequence[str] | None = None) -> dict[str, int]:
        target = tuple(names) if names is not None else STREAM_NAMES
        return {name: self.child_seed(name) for name in target if not name.startswith("cross.")}

    def used_streams(self) -> tuple[str, ...]:
        return self._registry.used_streams()

    def __repr__(self) -> str:  # pragma: no cover - デバッグ用
        return f"AssetStreamView(asset={self._asset_index}, registry={self._registry!r})"
