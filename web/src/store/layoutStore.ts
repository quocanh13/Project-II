import { create } from "zustand"
import type { Toast } from "../types"

interface ToastStore{
    toastList: Toast[],
    id: number

    addToast(
        title?: string, 
        message?: string,
        closeTime?: number,
        error?: boolean 
    ) : void

    removeToast(id: number) : void
}

export const useToastStore = create<ToastStore>((set, get)=>({
    id : 0,
    toastList : [],
    addToast(
        title = "", 
        message = "", 
        closeTime = 8000, 
        error = true
    ) {
        set((state)=>{
            const id = state.id
            const toast: Toast = {message, title, id, closeTime, error}
            if(closeTime != undefined)
                setTimeout(()=>{get().removeToast(id)}, closeTime)
            return {toastList : [...state.toastList, toast], id: state.id + 1}
        })

    },
    removeToast(id: number) {
        set(state => {
            return {toastList: state.toastList.filter((v) => v.id != id)}
        })
    },
}))