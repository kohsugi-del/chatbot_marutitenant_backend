from fastapi import (
    FastAPI,
    UploadFile,
    File as FastAPIFile,
    Depends,
    HTTPException,
    BackgroundTasks,
    Request,
    Response,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, shutil, re, uuid, logging, secrets
from dotenv import load_dotenv

load_dotenv(override=True)

from rag_core import answer
from vector_search import search
from agentic_rag import agentic_answer
from pipeline import advanced_agentic_answer

# =========================
# DB
# =========================
from database import SessionLocal, engine
from sqlalchemy import text
from sqlalchemy.orm import Session
from models_tenant import Tenant
from models_site import Site
from models_file import File as FileModel
from models_log import SessionLog, TurnLog
from schemas_tenant import TenantCreate, TenantUpdate, TenantResponse
from schemas_site import SiteCreate, SiteResponse, ReingestResponse
from schemas_file import FileResponse

# テーブル作成（新規のみ・既存テーブルは変更しない）
Tenant.metadata.create_all(bind=engine)
Site.metadata.create_all(bind=engine)
FileModel.metadata.create_all(bind=engine)
SessionLog.metadata.create_all(bind=engine)
TurnLog.metadata.create_all(bind=engine)


def _migrate_columns():
    """既存テーブルにカラムを追加（存在する場合はスキップ）"""
    stmts = [
        "ALTER TABLE sites ADD COLUMN tenant_id VARCHAR(36)",
        "ALTER TABLE files ADD COLUMN tenant_id VARCHAR(36)",
        "ALTER TABLE tenants ADD COLUMN client_id VARCHAR(64)",
        "ALTER TABLE tenants ADD COLUMN phone_normal VARCHAR",
        "ALTER TABLE tenants ADD COLUMN phone_emergency VARCHAR",
        "ALTER TABLE tenants ADD COLUMN business_hours VARCHAR",
        "ALTER TABLE tenants ADD COLUMN emergency_keywords TEXT",
        "ALTER TABLE tenants ADD COLUMN topic_keywords TEXT",
    ]
    with engine.connect() as conn:
        for stmt in stmts:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # カラムが既に存在する場合はスキップ


_migrate_columns()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# 管理者シークレット（テナントCRUD用）
# =========================
ADMIN_SECRET = (os.getenv("ADMIN_SECRET") or "").strip()


def require_admin(x_admin_secret: Optional[str] = Header(None, alias="X-Admin-Secret")):
    if not ADMIN_SECRET:
        raise HTTPException(status_code=503, detail="ADMIN_SECRET not configured")
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")


# =========================
# テナント認証依存性
# =========================
def get_tenant(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> Tenant:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    tenant = db.query(Tenant).filter(Tenant.api_key == x_api_key).first()
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


# =========================
# App
# =========================
app = FastAPI()

log = logging.getLogger("uvicorn.error")

# =========================
# CORS
# =========================
_CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "")
_CORS_ORIGINS = [o.strip() for o in _CORS_ORIGINS_ENV.split(",") if o.strip()]
if not _CORS_ORIGINS:
    _CORS_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.pages\.dev",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{rest_of_path:path}")
def preflight_handler(rest_of_path: str, request: Request):
    return Response(status_code=200)


@app.get("/__ping")
def ping():
    return {"ok": True}


# =========================
# Tenant CRUD（管理者のみ）
# =========================
@app.post("/tenants", response_model=TenantResponse, dependencies=[Depends(require_admin)])
def create_tenant(body: TenantCreate, db: Session = Depends(get_db)):
    api_key = body.api_key or secrets.token_urlsafe(32)
    if db.query(Tenant).filter(Tenant.api_key == api_key).first():
        raise HTTPException(status_code=409, detail="API key already exists")
    tenant = Tenant(name=body.name, system_prompt=body.system_prompt, api_key=api_key)
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


@app.get("/tenants", response_model=List[TenantResponse], dependencies=[Depends(require_admin)])
def list_tenants(db: Session = Depends(get_db)):
    return db.query(Tenant).order_by(Tenant.created_at).all()


@app.get("/tenants/{tenant_id}", response_model=TenantResponse, dependencies=[Depends(require_admin)])
def get_tenant_by_id(tenant_id: str, db: Session = Depends(get_db)):
    t = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return t


@app.patch("/tenants/{tenant_id}", response_model=TenantResponse, dependencies=[Depends(require_admin)])
def update_tenant(tenant_id: str, body: TenantUpdate, db: Session = Depends(get_db)):
    t = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    if body.name is not None:
        t.name = body.name
    if body.system_prompt is not None:
        t.system_prompt = body.system_prompt
    db.commit()
    db.refresh(t)
    return t


@app.delete("/tenants/{tenant_id}", dependencies=[Depends(require_admin)])
def delete_tenant(tenant_id: str, db: Session = Depends(get_db)):
    t = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Tenant not found")
    db.delete(t)
    db.commit()
    return {"status": "deleted"}


# =========================
# Chat API
# =========================
class ChatBody(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None
    top_k: int = 8
    session_id: Optional[str] = None


def get_question(body: ChatBody) -> str:
    q = (body.question or body.message or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question (or message) is required")
    return q


def build_refs(retrieved):
    refs = []
    for d, s in retrieved:
        try:
            src = d.get("source", "") if isinstance(d, dict) else ""
        except Exception:
            src = ""
        refs.append({"source": src, "score": float(s)})
    return refs


@app.post("/chat")
def chat(body: ChatBody, db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    q = get_question(body)

    session_id = body.session_id
    if not session_id:
        session_obj = SessionLog(municipality_id=tenant.id)
        db.add(session_obj)
        db.commit()
        db.refresh(session_obj)
        session_id = session_obj.id

    try:
        turn_order = db.query(TurnLog).filter(TurnLog.session_id == session_id).count() + 1
        db.add(TurnLog(session_id=session_id, turn_order=turn_order, role="user", content=q))
        db.commit()
    except Exception:
        log.exception("[log] failed to save user turn")

    retrieved = search(q, top_k=body.top_k, tenant_id=tenant.id)
    ans = answer(q, retrieved, system_prompt=tenant.system_prompt)
    refs = build_refs(retrieved)

    try:
        turn_order2 = db.query(TurnLog).filter(TurnLog.session_id == session_id).count() + 1
        db.add(TurnLog(session_id=session_id, turn_order=turn_order2, role="assistant", content=ans))
        db.commit()
    except Exception:
        log.exception("[log] failed to save assistant turn")

    return {"answer": ans, "references": refs, "session_id": session_id}


@app.post("/agentic_chat")
def agentic_chat(body: ChatBody, tenant: Tenant = Depends(get_tenant)):
    """Multi-LLM Agentic RAG: QueryRewrite → Search → Compress → Answer → Reflection"""
    q = get_question(body)
    result = advanced_agentic_answer(q, tenant_id=tenant.id, system_prompt=tenant.system_prompt)
    return result


@app.post("/ask")
def ask(body: ChatBody, tenant: Tenant = Depends(get_tenant)):
    q = get_question(body)
    retrieved = search(q, top_k=body.top_k, tenant_id=tenant.id)
    ans = answer(q, retrieved, system_prompt=tenant.system_prompt)
    refs = build_refs(retrieved)
    return {"answer": ans, "references": refs}


@app.post("/embed")
def embed(body: ChatBody, tenant: Tenant = Depends(get_tenant)):
    q = get_question(body)
    retrieved = search(q, top_k=body.top_k, tenant_id=tenant.id)
    ans = answer(q, retrieved, system_prompt=tenant.system_prompt)
    return {"answer": ans}


# =========================
# Sites（テナントスコープ）
# =========================
@app.post("/sites", response_model=SiteResponse)
def create_site(site: SiteCreate, db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    db_site = Site(
        tenant_id=tenant.id,
        url=site.url,
        scope=site.scope,
        type=site.type,
        status="pending",
    )
    db.add(db_site)
    db.commit()
    db.refresh(db_site)
    return db_site


@app.get("/sites", response_model=List[SiteResponse])
def list_sites(db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    return db.query(Site).filter(Site.tenant_id == tenant.id).order_by(Site.id.desc()).all()


@app.post("/sites/{site_id}/reingest", response_model=ReingestResponse)
def reingest_site(site_id: int, db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    site = db.query(Site).filter(Site.id == site_id, Site.tenant_id == tenant.id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    site.status = "pending"
    if hasattr(site, "error_message"):
        site.error_message = None  # type: ignore[attr-defined]
    db.commit()
    return {"status": "queued", "site_id": site.id}


@app.delete("/sites/{site_id}")
def delete_site(site_id: int, db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    site = db.query(Site).filter(Site.id == site_id, Site.tenant_id == tenant.id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    db.delete(site)
    db.commit()
    return {"status": "deleted"}


# =========================
# Files（テナントスコープ）
# =========================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(DATA_DIR, exist_ok=True)

_invalid_chars = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def safe_filename(original: str) -> str:
    name = os.path.basename(original or "").strip()
    if not name:
        name = "upload.pdf"
    name = _invalid_chars.sub("_", name)
    name = name.rstrip(". ").strip()
    if not name:
        name = "upload.pdf"
    return name


def unique_path(dir_path: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    uid = uuid.uuid4().hex[:8]
    return os.path.join(dir_path, f"{base}_{uid}{ext}")


@app.post("/files", response_model=FileResponse)
def upload_file(
    file: UploadFile = FastAPIFile(...),
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_tenant),
):
    try:
        fn = safe_filename(file.filename)
        save_path = unique_path(DATA_DIR, fn)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        db_file = FileModel(
            tenant_id=tenant.id,
            filename=os.path.basename(save_path),
            error_message=None,
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        return db_file

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload failed: {type(e).__name__}: {e}")


@app.get("/files")
def list_files(db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    rows = db.query(FileModel).filter(FileModel.tenant_id == tenant.id).order_by(FileModel.id.desc()).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "status": getattr(r, "status", "uploaded"),
            "ingested_chunks": getattr(r, "ingested_chunks", 0),
            "error_message": getattr(r, "error_message", None),
        }
        for r in rows
    ]


@app.post("/files/{file_id}/reingest")
def reingest_file(file_id: int, db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    f = db.query(FileModel).filter(FileModel.id == file_id, FileModel.tenant_id == tenant.id).first()
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    if hasattr(f, "status"):
        f.status = "pending"  # type: ignore[attr-defined]
    if hasattr(f, "error_message"):
        f.error_message = None  # type: ignore[attr-defined]
    db.commit()
    return {"status": "queued", "file_id": f.id}


@app.delete("/files/{file_id}")
def delete_file(file_id: int, db: Session = Depends(get_db), tenant: Tenant = Depends(get_tenant)):
    f = db.query(FileModel).filter(FileModel.id == file_id, FileModel.tenant_id == tenant.id).first()
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_path = os.path.join(DATA_DIR, f.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"remove failed: {type(e).__name__}: {e}")

    db.delete(f)
    db.commit()
    return {"status": "deleted"}


# =========================
# ingest 実行
# =========================
from ingest import ingest_site_from_db
from pdf_ingest import ingest_pdf


def _set_site_status(db: Session, site_id: int, status: str, error_message: Optional[str] = None):
    site = db.query(Site).filter(Site.id == site_id).first()
    if not site:
        return
    site.status = status
    if hasattr(site, "error_message"):
        site.error_message = error_message  # type: ignore[attr-defined]
    db.commit()


@app.post("/sites/{site_id}/reingest_local", response_model=ReingestResponse)
def reingest_local(
    site_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_tenant),
):
    site = db.query(Site).filter(Site.id == site_id, Site.tenant_id == tenant.id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    site.status = "crawling"
    if hasattr(site, "error_message"):
        site.error_message = None  # type: ignore[attr-defined]
    db.commit()

    def task(site_id_: int):
        db2 = SessionLocal()
        try:
            log.info(f"[ingest] start site_id={site_id_}")
            ingest_site_from_db(site_id_, max_pages=50, batch_size=5, sleep_sec=0.2, dry_run=False)
            log.info(f"[ingest] done site_id={site_id_}")
            _set_site_status(db2, site_id_, "done", None)
        except Exception as e:
            log.exception(f"[ingest] error site_id={site_id_}")
            _set_site_status(db2, site_id_, "error", f"{type(e).__name__}: {e}")
            raise
        finally:
            db2.close()

    background_tasks.add_task(task, site_id)
    return {"status": "queued", "site_id": site_id}


@app.post("/files/{file_id}/ingest_local")
def ingest_file_local(
    file_id: int,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(get_tenant),
):
    f = db.query(FileModel).filter(FileModel.id == file_id, FileModel.tenant_id == tenant.id).first()
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    pdf_path = os.path.join(DATA_DIR, f.filename)

    if hasattr(f, "status"):
        f.status = "crawling"  # type: ignore[attr-defined]
    if hasattr(f, "error_message"):
        f.error_message = None  # type: ignore[attr-defined]
    db.commit()

    try:
        result = ingest_pdf(
            file_id=file_id,
            pdf_path=pdf_path,
            file_name=f.filename,
            tenant_id=tenant.id,
        )

        if hasattr(f, "status"):
            f.status = "done"  # type: ignore[attr-defined]
        db.commit()

        return {"status": "done", "file_id": file_id, **result}

    except Exception as e:
        try:
            if hasattr(f, "status"):
                f.status = "error"  # type: ignore[attr-defined]
            if hasattr(f, "error_message"):
                f.error_message = f"{type(e).__name__}: {e}"  # type: ignore[attr-defined]
            db.commit()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"ingest failed: {type(e).__name__}: {e}")
