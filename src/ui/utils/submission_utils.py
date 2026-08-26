import io
import zipfile

def parse_btc_uploaded_files(uploaded_files):
    queries = []
    for file in uploaded_files:
        filename = file.name.lower()
        q_text = file.getvalue().decode("utf-8").strip()
        
        q_type = "kis"
        if "-qa" in filename: q_type = "qa"
        elif "-trake" in filename: q_type = "trake"

        q_id = file.name.rsplit('.', 1)[0]
        queries.append({"id": q_id, "type": q_type, "text": q_text})
    queries.sort(key=lambda x: x["id"])
    return queries

def create_zip_in_memory(cart_data):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for q_id, items in cart_data.items():
            csv_buffer = io.StringIO()
            for item in items:
                vid = item["video_id"]
                if "frames" in item: 
                    clean_frames = [str(f).replace("~", "").strip() for f in item["frames"]]
                    row_str = f"{vid}," + ",".join(clean_frames)
                else:
                    raw_frame = str(item["frame_idx"]).replace("~", "").strip()
                    if "answer" in item:
                        safe_ans = str(item["answer"]).replace('"', '""').strip()
                        row_str = f'{vid},{raw_frame},"{safe_ans}"'
                    else:
                        row_str = f"{vid},{raw_frame}"
                csv_buffer.write(row_str + "\n")
            zip_file.writestr(f"submission/{q_id}.csv", csv_buffer.getvalue().encode('utf-8'))
    return zip_buffer.getvalue()