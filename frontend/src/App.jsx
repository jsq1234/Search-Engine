import { useState } from "react";
import SearchBar from "./SearchBar";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState([]);

  const fetchDocuments = async () => {
    try {
      const backendUrl = "http://localhost:5000/api/search";
      const searchParams = new URLSearchParams();
      searchParams.set("query", query);
      searchParams.set("k", 10);
      const response = await fetch(`${backendUrl}?${searchParams.toString()}`);
      if (!response.ok) {
        throw new Error(`HTTP error! Status : ${response.ok}`);
      }
      const responseData = await response.json();
      setResult(responseData);
      window.history.pushState({}, "", `?${searchParams.toString()}`);
    } catch (error) {
      console.log("Error during POST request:", error);
    }
  };

  return (
    <>
      {result.length == 0 && (
        <h1
          className="text-[6rem] md:text-[10rem] font-serif xl:w-[70%] 
        mx-auto xl:rounded-full dark:text-blue-100 
        md:mt-[8rem] mt-[4rem] text-center bg-[#000000]"
        >
          Search
        </h1>
      )}

      <SearchBar
        queryVar={query}
        setQuery={setQuery}
        onSubmit={fetchDocuments}
      />

      <div className="flex flex-col gap-4 mt-5">
        {result.length != 0 &&
          result.map((elem, index) => (
            <div
              key={index}
              className="w-[75%] mx-auto border-4 rounded-xl p-3 pr-5 pl-5 hover:bg-[#156161] hover:shadow-2xl"
            >
              <div className="flex items-center justify-between gap-4">
                <h1 className="text-2xl font-semibold">{elem.title}</h1>
                <h1 className="text-xl font-semibold">[{elem.score}]</h1>
              </div>
              <p className="text-xl mt-3">{elem.body}</p>
            </div>
          ))}
        v
      </div>
    </>
  );
}

export default App;
