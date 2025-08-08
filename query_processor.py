from typing import List, Tuple
from collections import defaultdict

class ResumeMatcher:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def get_top_resume_ids_from_chunks(self, job_description: str, k=10) -> List[Tuple[str, float]]:
        # Step 1: Search top-k similar chunks to job description
        results = self.vectorstore.similarity_search_with_score(job_description, k=k)

        # Step 2: Collect resume IDs and all their similarity scores
        id_scores = defaultdict(list)
        for doc, score in results:
            resume_id = doc.metadata.get("ID")
            if resume_id:
                id_scores[resume_id].append(score)

        # Step 3: Sort resume IDs by their best (lowest) similarity score
        sorted_ids = sorted(id_scores.items(), key=lambda x: min(x[1]))

        # Return a list of tuples: (resume_id, best_similarity_score)
        return [(rid, min(scores)) for rid, scores in sorted_ids]

    def get_full_resume_by_id(self, resume_id: str) -> str:
        # Pull all chunks related to this resume
        docs = self.vectorstore.similarity_search(resume_id, k=100)
        chunks = [doc.page_content for doc in docs if doc.metadata.get("ID") == resume_id]
        print(chunks)
        return "\n".join(chunks)
        
    def evaluate_resume_against_jd(self, resume_text: str, job_description: str) -> str:
        prompt = f"""
            You are a senior technical recruiter with deep expertise in evaluating software engineering talent.

            Your task is to evaluate a candidate's resume against a job description and return a structured analysis in **valid JSON format**.

            ---

            **Job Description**:
            {job_description}

            **Candidate Resume**:
            {resume_text}

            ---

            **Instructions**:
            1. Identify the **top 2-3 most relevant matching criteria** between the job description and resume.
            2. For each criterion:
            - Clearly name the criterion (e.g., "Python", "System Design", etc.)
            - Give a **score out of 10**
            - Provide a **justification** (1-2 sentences)

            3. Provide an overall match score out of 10.
            4. Summarize the match in 2-3 sentences.

            ---

            Important : Return a **pure JSON object** using this exact structure — do not include markdown or Backticks or text before or after:

            json:
            
            "criteria": [
                {{
                "name": "Criterion Name",
                "score": 0,
                "justification": "Reason why the score was given"
                }},
                {{
                "name": "Criterion Name",
                "score": 0,
                "justification": "Reason why the score was given"
                }}
            ],
            "overall_score": 0,
            "summary": "Brief 2-3 sentence summary explaining how well the candidate fits the role."
            
            
            """

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def run_pipeline(self, job_description: str, top_k=10):
        print("here")
        top_resume_ids_with_scores = self.get_top_resume_ids_from_chunks(job_description, k=top_k)
        results = []

        for resume_id, similarity_score in top_resume_ids_with_scores:
            resume_text = self.get_full_resume_by_id(resume_id)
            evaluation = self.evaluate_resume_against_jd(resume_text, job_description)
            results.append({
                "resume_id": resume_id,
                "cosine_similarity": round(1 - similarity_score, 4),  # convert distance to similarity
                "evaluation": evaluation
            })

        return results

    