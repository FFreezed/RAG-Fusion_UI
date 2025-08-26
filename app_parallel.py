import streamlit as st
import rag_pipeline_parallel  # Mengimpor file pipeline paralel
import pandas as pd
import os

# Mengatur konfigurasi halaman
st.set_page_config(layout="wide")

# Mengambil kunci API dari Streamlit secrets
OPENROUTER_API_KEY =  os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("PERSONAL_2")

# --- UI Aplikasi ---
st.title("Aplikasi Deteksi Hoaks Berbasis RAG Fusion")
st.markdown(
    "Masukkan klaim berita di bawah untuk melakukan pengecekan hoaks otomatis.")

# Area input pengguna
user_input = st.text_area(
    "Masukkan klaim yang ingin diperiksa:",
    height=100,
    placeholder="Contoh: Pemerintah akan memberikan bantuan sosial Rp 5 juta untuk setiap keluarga."
)

# Tombol untuk memulai proses
if st.button("Lakukan Pengecekan Hoaks"):
    # Validasi input dan ketersediaan kunci API
    if not user_input.strip():
        st.warning("Mohon masukkan klaim terlebih dahulu.")
    elif not OPENROUTER_API_KEY or not TAVILY_API_KEY:
        st.error("Kunci API untuk OpenRouter atau Tavily tidak ditemukan. Harap konfigurasikan di Streamlit secrets.")
    else:
        # Menjalankan pipeline dengan spinner
        with st.spinner("Memproses klaim Anda... Ini mungkin memakan waktu beberapa saat."):
            # Memanggil fungsi pipeline paralel
            pipeline_results, error = rag_pipeline_parallel.process_rag_pipeline_parallel(
                user_input,
                openrouter_api_key=OPENROUTER_API_KEY,
                tavily_api_key=TAVILY_API_KEY
            )

        # Menangani hasil atau error dari pipeline
        if error:
            st.error(f"Terjadi kesalahan selama proses pipeline: {error}")
        else:
            st.success(f"Pengecekan Hoaks Selesai! Total waktu: {pipeline_results.get('total_time', 'N/A')} detik.")

            # --- Menampilkan Hasil Secara Vertikal ---

            # Menghapus 'col1, col2 = st.columns(2)' dan blok 'with'

            # st.subheader("1. Klaim Pengguna")
            # st.write(pipeline_results.get("user_input", "N/A"))

            # st.subheader("2. Kueri yang Dihasilkan")
            # queries = pipeline_results.get("queries", [])
            # if queries:
            #     for i, query in enumerate(queries, 1):
            #         st.write(f"- {query}")
            # else:
            #     st.write("Tidak ada kueri yang dihasilkan.")

            # st.subheader("3. Statistik Pencarian")
            # st.info(f"Total hasil pencarian awal: {pipeline_results.get('search_results_count', 0)}")
            # processed_stats = pipeline_results.get("processed_results", {}).get("stats", {})
            # st.info(f"Hasil yang diproses (unik): {processed_stats.get('total_processed_results', 0)}")
            # st.info(f"Konten diekstrak dari URL: {processed_stats.get('urls_extracted_count', 0)}")

            # st.subheader("4. Analisis Pengecekan Hoaks")
            st.subheader("Analisis Pengecekan Hoaks")
            fact_check_result = pipeline_results.get("fact_check_analysis", {})
            if fact_check_result.get("status") == "success":
                st.markdown(fact_check_result.get("analysis", "Tidak ada analisis yang dihasilkan."))
            else:
                st.error(f"Gagal menghasilkan analisis: {fact_check_result.get('analysis', 'Error tidak diketahui.')}")

            st.subheader("Statistik Pencarian")
            st.info(f"Total hasil pencarian awal: {pipeline_results.get('search_results_count', 0)}")
            processed_stats = pipeline_results.get("processed_results", {}).get("stats", {})
            st.info(f"Hasil yang diproses (unik): {processed_stats.get('total_processed_results', 0)}")
            st.info(f"Konten diekstrak dari URL: {processed_stats.get('urls_extracted_count', 0)}")

            # st.subheader("5. Sumber Bukti yang Diproses")
            st.subheader("Sumber Bukti yang Diproses")
            processed_results = pipeline_results.get("processed_results", {})
            results_data = []
            for res in processed_results.get('results', []):
                content_preview = (res.get('extracted_content') or res.get('raw_content') or res.get(
                    'content') or "Tidak ada konten.")
                results_data.append({
                    "Judul": res.get('title', 'N/A'),
                    "URL": res.get('url', 'N/A'),
                    "Domain": res.get('domain', 'N/A'),
                    "Skor Relevansi": res.get('score', 0),
                    "Konten (Preview)": content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
                })

            if results_data:
                df_results = pd.DataFrame(results_data)
                st.dataframe(df_results, use_container_width=True)
            else:
                st.warning("Tidak ada sumber bukti yang relevan ditemukan atau diproses.")