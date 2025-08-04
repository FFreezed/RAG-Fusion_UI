import streamlit as st
import rag_pipeline
import pandas as pd
import os

st.set_page_config(layout="wide")

st.title("RAG-Fusion Based Hoax Detection")
st.markdown("Masukkan klaim berita di bawah untuk melakukan deteksi hoaks otomatis.")

user_input = st.text_area("Masukkan klaim yang ingin di check:", height=100,
                          placeholder="Contoh: Jokowi mengundurkan diri dari posisi presiden")

if st.button("Lakukan Deteksi Hoaks"):
    if user_input:
        with st.spinner("Memproses klaim, mohon tunggu..."):
            try:
                pipeline_results = rag_pipeline.process_output(user_input)

                st.success("Deteksi Hoaks Selesai!")

                st.subheader("1. Klaim Pengguna:")
                st.write(pipeline_results["user_input"])

                st.subheader("2. Kueri yang Dihasilkan:")
                for i, query in enumerate(pipeline_results["queries"], 1):
                    st.write(f"- {query}")

                st.subheader("3. Hasil Pencarian Awal (Tavily):")
                st.info(f"Total hasil pencarian awal: {pipeline_results['search_results_count']}")

                st.subheader("4. Hasil Pencarian Diproses & Ekstraksi Konten:")
                processed_results = pipeline_results["processed_results"]
                st.info(f"Total hasil yang diproses (setelah deduplikasi & top N): {processed_results['stats']['total_processed_results']}")

                # Display processed results in a table for better readability
                results_data = []
                for res in processed_results['results']:
                    content_preview = (res.get('extracted_content') or res.get('raw_content') or res.get('content') or "Tidak ada konten yang tersedia.")
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
                    st.warning("Tidak ada hasil pencarian yang relevan ditemukan.")


                st.subheader("5. Analisis:")
                fact_check_result = pipeline_results["fact_check_analysis"]
                if fact_check_result["status"] == "success":
                    st.markdown(fact_check_result["analysis"])
                else:
                    st.error(f"Terjadi kesalahan saat menghasilkan analisis: {fact_check_result['analysis']}")

                st.markdown("---")
                st.write("Analisis lengkap juga disimpan di `./output/fact_check_analysis.md`.")
                st.write("Prompt lengkap yang dikirim ke LLM disimpan di `./output/full_prompt.txt`.")
                st.write("Hasil pencarian awal dan yang diproses disimpan di `./output/initial_search_results_grouped.json` dan `./output/processed_search_results.json`.")

            except Exception as e:
                st.error(f"Terjadi kesalahan tak terduga selama proses: {e}")
                st.exception(e) # Menampilkan detail error untuk debugging
    else:
        st.warning("Mohon masukkan klaim terlebih dahulu.")